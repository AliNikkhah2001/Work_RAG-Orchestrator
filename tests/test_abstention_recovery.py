"""Post-abstention recovery when retrieval evidence is already present."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from work_rag_orchestrator.content_registry import abstention_phrase, refusal_message
from work_rag_orchestrator.nodes.format_response import (
    _has_retrieval_evidence,
    format_response,
)
from work_rag_orchestrator.state import RAGState


class TestAbstentionRecovery:
    @pytest.mark.asyncio
    async def test_h1b_regenerates_when_broaden_used_and_chunks_present(self):
        """broaden_used must not block H1b regen when chunks exist."""
        abstain = refusal_message("no_evidence")
        recovered = "پاسخ بازسازی‌شده بر اساس شواهد موجود."
        state: RAGState = {
            "request_id": "abstention-recovery-01",
            "messages": [],
            "query": "سوال تست",
            "guardrail_decision": {"allowed": True},
            "retrieved_chunks": [
                {
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "title": "عنوان",
                    "heading": "بخش",
                    "content": "متن شواهد مرتبط با سوال.",
                    "score": 0.85,
                    "rerank_score": 0.85,
                }
            ],
            "prompt_messages": [],
            "answer": abstain,
            "citations": [],
            "error": None,
            "blocked": False,
            "refusal_message": None,
            "broaden_used": True,
        }

        async def _fake_regen(s: RAGState) -> RAGState:
            s = dict(s)
            s["answer"] = recovered
            return s

        with patch(
            "work_rag_orchestrator.nodes.format_response._regenerate_with_existing_chunks",
            new=AsyncMock(side_effect=_fake_regen),
        ) as mock_regen, patch(
            "work_rag_orchestrator.nodes.format_response._broaden_retrieve_and_regenerate",
            new=AsyncMock(),
        ) as mock_broaden:
            result = await format_response(state)

        mock_regen.assert_awaited_once()
        mock_broaden.assert_not_awaited()
        assert result["broaden_after_abstention"] is True
        assert result["formatted_response"]["content"] == recovered
        assert len(result["formatted_response"]["citations"]) >= 1

    def test_has_retrieval_evidence_supplemental(self):
        assert _has_retrieval_evidence({"retrieved_chunks": []}, []) is False
        assert _has_retrieval_evidence(
            {"retrieved_chunks": [], "supplemental": [{"name": "x"}]}, []
        )
        assert abstention_phrase() in refusal_message("no_evidence")