"""Stateless query rewrite — resolve coreference using conversation history.

No DB, no checkpointer: WebUI already sends the full ``messages`` history on
every request, so the rewrite is a pure function of the incoming messages.

- 0-1 user turns: return the last user message unchanged (no LLM call).
- 2+ user turns: ask the configured upstream utility LLM (``UPSTREAM_LLM_BASE_URL``,
  same endpoint compaction uses) to rewrite the last user message into a
  standalone question, then strip quotes/whitespace.
- On ANY error/timeout: fall back to the raw last user message (never raise).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

log = logging.getLogger(__name__)

# Legacy defaults — used only when settings are unavailable. Prefer
# ``_rewrite_endpoint()`` so Docker/GPU orch hits litellm/manager, not
# container-local 127.0.0.1:9000 (hardcoded URL made every multi-turn rewrite
# and left long threads without coreference resolution on GPU).
REWRITE_URL = "http://127.0.0.1:9000/v1/chat/completions"
REWRITE_MODEL = "gemma-4-31b-triton-trt"
REWRITE_TIMEOUT_S = 60.0
REWRITE_MAX_TOKENS = 128


def _rewrite_endpoint() -> Tuple[str, str]:
    """Return (chat_completions_url, model) from orchestrator settings."""
    try:
        from .config import get_settings

        settings = get_settings()
        base = (settings.upstream_llm_base_url or "http://127.0.0.1:9000/v1").rstrip("/")
        url = base + "/chat/completions"
        model = settings.upstream_llm_model or REWRITE_MODEL
        return url, model
    except Exception:
        return REWRITE_URL, REWRITE_MODEL

# Last-N exchanges kept in prompts/spans; history block budget (chars).
HISTORY_EXCHANGES = 2
HISTORY_MAX_CHARS = 1500

_QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("«", "»"), ("“", "”"), ("‘", "’"))


def user_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return only non-empty user turns, in order."""
    return [m for m in (messages or []) if m.get("role") == "user" and (m.get("content") or "").strip()]


def last_exchanges_text(
    messages: List[Dict[str, str]],
    n_exchanges: int = HISTORY_EXCHANGES,
    max_chars: int = HISTORY_MAX_CHARS,
    include_current: bool = True,
) -> str:
    """Format the last N user+assistant exchanges as plain text.

    Each turn is one line (``کاربر: ...`` / ``دستیار: ...``), oldest first,
    truncated from the front so the result is at most ``max_chars`` chars.
    With ``include_current=False`` the trailing user turn (the question being
    answered) is excluded — for prompt history blocks.
    """
    turns = [m for m in (messages or []) if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
    if not include_current and turns and turns[-1].get("role") == "user":
        turns = turns[:-1]
    # Last N exchanges ≈ last 2*N turns (one user + one assistant each).
    turns = turns[-2 * n_exchanges :]
    lines = []
    for m in turns:
        label = "کاربر" if m.get("role") == "user" else "دستیار"
        text = " ".join((m.get("content") or "").split())
        lines.append(f"{label}: {text}")
    excerpt = "\n".join(lines)
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:]
    return excerpt


def build_rewrite_prompt(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build the LLM messages for rewriting the last user turn.

    System + user templates come from the content registry (KB prompts pack).
    """
    from .content_registry import require_prompt

    users = user_messages(messages)
    current = users[-1].get("content", "") if users else ""
    history = last_exchanges_text(messages, include_current=False)
    system = require_prompt("rewrite_system")
    if history:
        user_content = require_prompt(
            "rewrite_user_with_history", history=history, current=current
        )
    else:
        user_content = require_prompt("rewrite_user_no_history", current=current)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def clean_rewritten(text: str) -> str:
    """Strip surrounding whitespace and matching quote pairs."""
    cleaned = (text or "").strip()
    for _ in range(3):  # nested quotes at most
        stripped = cleaned.strip()
        for left, right in _QUOTE_PAIRS:
            if len(stripped) >= 2 and stripped.startswith(left) and stripped.endswith(right):
                stripped = stripped[len(left) : -len(right)].strip()
                break
        else:
            break
        cleaned = stripped
    return cleaned


async def rewrite_query(
    messages: List[Dict[str, str]],
    httpx_client: Optional[Any] = None,
) -> str:
    """Rewrite the last user message into a standalone question.

    Returns the raw last user message unchanged when there are 0-1 user
    turns, when the rewrite LLM errors/times out, or when its output is
    empty after cleaning. Never raises.
    """
    users = user_messages(messages)
    if not users:
        return ""
    raw_last = users[-1].get("content", "")
    if len(users) <= 1:
        return raw_last
    try:
        prompt = build_rewrite_prompt(messages)
        rewrite_url, rewrite_model = _rewrite_endpoint()
        payload = {
            "model": rewrite_model,
            "messages": prompt,
            "temperature": 0,
            "max_tokens": REWRITE_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if httpx_client is not None:
            resp = await httpx_client.post(rewrite_url, json=payload, timeout=REWRITE_TIMEOUT_S)
        else:
            async with httpx.AsyncClient(timeout=REWRITE_TIMEOUT_S) as client:
                resp = await client.post(rewrite_url, json=payload, timeout=REWRITE_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        cleaned = clean_rewritten(content)
        if not cleaned:
            log.warning("query rewrite returned empty output; falling back to raw query")
            return raw_last
        log.info("query rewrite raw=%r rewritten=%r", raw_last[:120], cleaned[:120])
        return cleaned
    except Exception as e:
        log.warning("query rewrite failed (%s); falling back to raw query", e)
        return raw_last
