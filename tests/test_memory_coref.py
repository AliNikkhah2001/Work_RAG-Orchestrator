"""Regression tests for stateless conversation memory (coreference rewrite).

3-turn scenario (Persian):
  turn1 user: «مدیر شرکت اعتبارسنجی کیه»
  turn2 user: «رضا قاسم بور کیه» + assistant answer naming Reza Ghasempour
  turn3 user: «دکتری کجا بوده حالا»  (elliptical — refers to Reza Ghasempour)

The rewrite must resolve the reference so the standalone question contains
the name. The LLM call is ALWAYS mocked — these tests run fully offline and
must never hit http://127.0.0.1:18000.
"""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from work_rag_orchestrator.rewrite import (
    REWRITE_URL,
    REWRITE_MODEL,
    build_rewrite_prompt,
    clean_rewritten,
    last_exchanges_text,
    rewrite_query,
)

TURN1 = "مدیر شرکت اعتبارسنجی کیه"
TURN2 = "رضا قاسم بور کیه"
ASSISTANT2 = "رضا قاسم‌پور مدیرعامل شرکت اعتبارسنجی ایران است."
TURN3 = "دکتری کجا بوده حالا"

MESSAGES_3TURN = [
    {"role": "user", "content": TURN1},
    {"role": "assistant", "content": "مدیرعامل شرکت اعتبارسنجی ایران آقای رضا قاسم‌پور است."},
    {"role": "user", "content": TURN2},
    {"role": "assistant", "content": ASSISTANT2},
    {"role": "user", "content": TURN3},
]


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeClient:
    """Stand-in for httpx.AsyncClient; records the request, returns content."""

    def __init__(self, content: str, captured: dict):
        self._content = content
        self._captured = captured

    async def post(self, url, json=None, timeout=None):
        self._captured["url"] = url
        self._captured["payload"] = json
        self._captured["timeout"] = timeout
        return _FakeResp(self._content)


class _ExplodingClient:
    """Raises if .post is ever called (proves no HTTP for 0-1 turns)."""

    async def post(self, *a, **k):
        raise AssertionError("no HTTP call expected for 0-1 user turns")


class _ErrorClient:
    async def post(self, *a, **k):
        raise httpx.ConnectError("connection refused")


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_single_turn_returns_unchanged_without_http(self):
        out = await rewrite_query(
            [{"role": "user", "content": TURN1}], httpx_client=_ExplodingClient()
        )
        assert out == TURN1

    @pytest.mark.asyncio
    async def test_no_messages_returns_empty(self):
        assert await rewrite_query([], httpx_client=_ExplodingClient()) == ""

    @pytest.mark.asyncio
    async def test_coref_rewrite_contains_name(self):
        captured: dict = {}
        out = await rewrite_query(
            MESSAGES_3TURN,
            httpx_client=_FakeClient(
                '"رضا قاسم‌پور دکتری خود را کجا گذرانده است؟"', captured
            ),
        )
        assert "قاسم" in out
        # quotes stripped
        assert not out.startswith('"') and not out.endswith('"')
        # request went to direct llama-server with expected shape
        assert captured["url"] == REWRITE_URL
        payload = captured["payload"]
        assert payload["model"] == REWRITE_MODEL
        assert payload["temperature"] == 0
        assert payload["max_tokens"] == 128
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert captured["timeout"] == 60.0

    @pytest.mark.asyncio
    async def test_prompt_building_includes_history(self):
        prompt = build_rewrite_prompt(MESSAGES_3TURN)
        assert prompt[0]["role"] == "system"
        # Persian instruction: standalone question, no greeting/explanation
        assert "فقط" in prompt[0]["content"] and "سلام" in prompt[0]["content"]
        body = prompt[1]["content"]
        assert "قاسم‌پور" in body  # assistant answer carried into history
        assert TURN3 in body  # current question present

    @pytest.mark.asyncio
    async def test_fallback_on_error_returns_raw(self):
        out = await rewrite_query(MESSAGES_3TURN, httpx_client=_ErrorClient())
        assert out == TURN3  # never raises, raw last message preserved

    @pytest.mark.asyncio
    async def test_fallback_on_empty_output(self):
        captured: dict = {}
        out = await rewrite_query(
            MESSAGES_3TURN, httpx_client=_FakeClient("   ", captured)
        )
        assert out == TURN3

    def test_clean_rewritten_strips_quotes(self):
        assert clean_rewritten('  "سلام"  ') == "سلام"
        assert clean_rewritten("«رضا قاسم‌پور کیست؟»") == "رضا قاسم‌پور کیست؟"

    def test_history_excerpt_bounded(self):
        excerpt = last_exchanges_text(MESSAGES_3TURN, include_current=True)
        assert "قاسم" in excerpt and TURN3 in excerpt
        assert len(excerpt) <= 1500


class TestRetrieveUsesRewrite:
    @pytest.mark.asyncio
    async def test_retrieve_searches_rewritten_query(self):
        from work_rag_orchestrator.nodes.retrieve import retrieve

        rewritten = "رضا قاسم‌پور دکتری خود را کجا گذرانده است؟"
        state = {
            "request_id": "mem-1",
            "messages": MESSAGES_3TURN,
            "query": TURN3,  # original kept for audit
            "rewritten_query": "",
            "guardrail_decision": {"allowed": True},
            "retrieved_chunks": [],
            "prompt_messages": [],
            "answer": "",
            "citations": [],
            "error": None,
            "blocked": False,
            "refusal_message": None,
        }
        with (
            patch(
                "work_rag_orchestrator.nodes.retrieve.rewrite_query",
                new=AsyncMock(return_value=rewritten),
            ),
            patch(
                "work_rag_orchestrator.nodes.retrieve.KnowledgebaseClient"
            ) as mock_kb,
            patch("work_rag_orchestrator.tracing.trace_span") as mock_span,
        ):
            mock_client = AsyncMock()
            mock_client.retrieve.return_value = []
            mock_kb.return_value.__aenter__.return_value = mock_client
            result = await retrieve(state)

        assert result["rewritten_query"] == rewritten
        assert result["query"] == TURN3  # original preserved
        # KB searched with the rewritten (normalized) query, not the raw one
        searched = mock_client.retrieve.call_args[0][0]
        assert "قاسم" in searched and searched != TURN3
        # both spans emitted: query-rewrite + retrieve (with history)
        span_names = [c[0][1] for c in mock_span.call_args_list]
        assert "query-rewrite" in span_names and "retrieve" in span_names


class TestBuildContextHistory:
    @pytest.mark.asyncio
    async def test_history_block_before_question(self):
        from work_rag_orchestrator.nodes.build_context import (
            build_context,
            MAX_CONTEXT_CHARS,
        )

        state = {
            "request_id": "mem-2",
            "messages": MESSAGES_3TURN,
            "query": TURN3,
            "rewritten_query": "رضا قاسم‌پور دکتری خود را کجا گذرانده است؟",
            "guardrail_decision": {"allowed": True},
            "retrieved_chunks": [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "title": "T",
                    "heading": "H",
                    "content": "رضا قاسم‌پور دارای دکتری است.",
                    "score": 0.9,
                }
            ],
            "prompt_messages": [],
            "answer": "",
            "citations": [],
            "error": None,
            "blocked": False,
            "refusal_message": None,
        }
        with patch("work_rag_orchestrator.tracing.trace_span"):
            result = await build_context(state)
        user_content = result["prompt_messages"][1]["content"]
        assert "[Conversation history]" in user_content
        assert "قاسم" in user_content  # prior exchange visible to generator
        hist_start = user_content.index("[Conversation history]")
        q_start = user_content.index("Question:")
        assert hist_start < q_start  # history BEFORE the question
        history_block = user_content[hist_start:q_start]
        assert len(history_block) <= 1500 + len("[Conversation history]\n")
        assert len(user_content) <= MAX_CONTEXT_CHARS + len("\n...[truncated]")

    @pytest.mark.asyncio
    async def test_new_chat_greeting_preserved_without_history(self):
        from work_rag_orchestrator.nodes.build_context import build_context

        state = {
            "request_id": "mem-3",
            "messages": [{"role": "user", "content": "سلام"}],
            "query": "سلام",
            "rewritten_query": "سلام",
            "guardrail_decision": {"allowed": True},
            "retrieved_chunks": [],
            "prompt_messages": [],
            "answer": "",
            "citations": [],
            "error": None,
            "blocked": False,
            "refusal_message": None,
        }
        with patch("work_rag_orchestrator.tracing.trace_span"):
            result = await build_context(state)
        system = result["prompt_messages"][0]["content"]
        assert "سلام" in system  # new-chat greeting logic kept
        assert "[Conversation history]" not in result["prompt_messages"][1]["content"]
