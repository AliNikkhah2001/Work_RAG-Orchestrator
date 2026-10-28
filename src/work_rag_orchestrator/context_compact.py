"""Rolling-summary context compaction ("auto compacting") for long threads.

The upstream model window is 8192 tokens (TRT-LLM rejects longer prompts with
HTTP 400 ``default_max_tokens`` overflow). When the conversation history would
push the assembled generation prompt past the token budget, older turns are
replaced by ONE LLM-generated rolling summary while the newest turns are kept
verbatim. Short sessions never trigger the summarizer (zero added latency).

Design notes:
- Stateless-friendly: the summary cache is keyed by the SHA-256 of the old-turn
  transcript, so identical history prefixes (WebUI re-sends full history each
  turn) hit the cache instead of re-summarizing.
- Failure policy: on ANY summarizer error return the original messages
  (compaction is an optimization, never a correctness dependency). The
  Guardrails ``_fit_upstream_messages`` trim remains the last-resort guard.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Tuple

from .config import get_settings

log = logging.getLogger(__name__)

# Empirical Persian-heavy char→token ratio for Gemma (observed: 72k chars ≈
# 16k tokens on the TRT-LLM upstream → ~4.5 chars/token).
CHARS_PER_TOKEN = 4.5

# Cache: transcript-hash -> summary text. Bounded to avoid unbounded growth.
_SUMMARY_CACHE: Dict[str, str] = {}
_CACHE_MAX = 128

# Rolling base for incremental re-summarization: when a long session grows by
# a couple of turns, the new transcript is the previous one plus a small tail.
# Re-summarizing the whole transcript every turn costs ~2-3s; instead keep the
# last (transcript, summary) and summarize only "previous summary + new turns".
# The fingerprint (hash of the transcript's first 200 chars) prevents mixing
# summaries across different sessions.
_BASE: Dict[str, Any] = {}

_INCREMENTAL_USER_TMPL = (
    "خلاصهٔ فعلی گفتگو:\n{old_summary}\n\n"
    "نوبت‌های جدید گفتگو:\n{new_turns}\n\n"
    "خلاصهٔ به‌روز و فشردهٔ کل گفتگو را در حداکثر ۶ جملهٔ فارسی بنویس "
    "(نکات مهم خلاصهٔ قبلی + نکات نوبت‌های جدید). فقط متن خلاصه را بنویس."
)

_SUMMARY_SYSTEM_PROMPT = (
    "تو خلاصه‌ساز گفتگو هستی. با توجه به تاریخچهٔ زیر، خلاصهٔ فشرده و کاربردی "
    "گفتگو را در حداکثر ۶ جملهٔ فارسی بنویس. این نکات را حفظ کن: موضوع و نوع "
    "موجودیت (شخص حقیقی یا شرکت) که کاربر درباره‌اش پرسیده، نام‌ها و کدهای "
    "ذکرشده، سوالاتی که پاسخ داده شده و جواب کوتاهشان، و هر سوال بی‌پاسخ. "
    "فقط متن خلاصه را بنویس — بدون مقدمه، بدون عنوان، بدون بولت."
)


def _norm_text(text: str) -> str:
    """Collapse whitespace for transcript/summary hygiene."""
    return " ".join((text or "").split())


def _turns(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only user/assistant turns with non-empty content, in order."""
    return [
        m
        for m in (messages or [])
        if m.get("role") in ("user", "assistant") and _norm_text(m.get("content") or "")
    ]


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate for a message list (chars / CHARS_PER_TOKEN)."""
    return int(sum(len(m.get("content") or "") for m in messages) / CHARS_PER_TOKEN)


def _transcript(old_turns: List[Dict[str, Any]]) -> str:
    """Build the summarizer input; cap each message AND the total so the
    summarizer prompt itself always fits the upstream window (repetitive
    Persian can tokenize as dense as ~3 chars/token). Oldest lines are
    dropped first — the newest old-turns matter most for continuity."""
    settings = get_settings()
    cap = max(settings.compact_msg_cap_chars, 200)
    total_cap = max(settings.compact_transcript_chars, cap * 2)
    lines = []
    for m in old_turns:
        label = "کاربر" if m.get("role") == "user" else "دستیار"
        text = _norm_text(m.get("content") or "")
        if len(text) > cap:
            text = text[:cap] + "…"
        lines.append(f"{label}: {text}")
    while len("\n".join(lines)) > total_cap and len(lines) > 1:
        lines.pop(0)
    return "\n".join(lines)


async def _summarize(transcript: str) -> str:
    """Call the utility LLM (direct OpenAI-compatible endpoint, no rails).

    One retry on transport errors / transient 5xx — the manager round-robins
    over several llama.cpp backends and a single dead backend must not disable
    compaction for the turn.
    """
    import asyncio
    import httpx

    settings = get_settings()
    url = settings.upstream_llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.upstream_llm_model,
        "messages": [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "max_tokens": settings.compact_summary_max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {settings.upstream_llm_api_key}"}
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(settings.compact_llm_timeout_seconds, connect=5.0),
                trust_env=False,
            ) as client:
                resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"upstream {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            data = resp.json()
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            summary = _norm_text(content)
            # Gemma occasionally prefixes meta text — strip a leading label line.
            summary = re.sub(r"^(خلاصه|Summary)\s*[:：]\s*", "", summary)
            max_chars = max(settings.compact_max_summary_chars, 200)
            if len(summary) > max_chars:
                summary = summary[:max_chars] + "…"
            return summary
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last_exc = e
            if attempt == 1:
                await asyncio.sleep(0.5)
    raise last_exc  # type: ignore[misc]


def compact_history_sync(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str, Dict[str, Any]] | None:
    """Synchronous part of compaction: threshold check + split + cache lookup.

    Returns ``None`` when compaction should not run (disabled, below threshold,
    nothing to compact). Otherwise returns
    ``(old_turns, recent_turns, transcript, cache_key, info)``.
    """
    settings = get_settings()
    info: Dict[str, Any] = {
        "compacted": False,
        "old_turns": 0,
        "cached": False,
        "summary_chars": 0,
        "estimate_tokens": 0,
    }
    if not settings.compact_enabled:
        return None
    turns = _turns(messages)
    if not turns:
        return None
    info["estimate_tokens"] = estimate_tokens(turns)
    budget_chars = settings.compact_threshold_tokens * CHARS_PER_TOKEN
    if sum(len(m.get("content") or "") for m in turns) <= budget_chars:
        return None

    recent_n = max(settings.compact_recent_turns, 2)
    if len(turns) <= recent_n:
        # Everything is "recent" — cannot compact without losing content.
        return None
    old = turns[:-recent_n]
    recent = turns[-recent_n:]

    transcript = _transcript(old)
    key = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    return old, recent, transcript, key, info


async def compact_history(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compact long histories into [summary] + recent turns.

    Returns ``(compacted_messages, info)``. ``info["compacted"]`` tells the
    caller whether a summary block was produced; ``info["summary"]`` holds the
    summary text when compacted. On any error the original messages are
    returned untouched (compaction must never break generation).
    """
    try:
        split = compact_history_sync(messages)
        if split is None:
            return messages, {
                "compacted": False, "old_turns": 0, "cached": False,
                "summary_chars": 0,
                "estimate_tokens": estimate_tokens(_turns(messages)),
            }
        old, recent, transcript, key, info = split
        cached = _SUMMARY_CACHE.get(key)
        if cached:
            info.update(compacted=True, old_turns=len(old), cached=True,
                        summary_chars=len(cached))
            compacted = [{"role": "system", "content": f"[خلاصهٔ گفتگوی پیشین]\n{cached}"}] + recent
            return compacted, info
        # Incremental fast path: same session grew by a few turns — summarize
        # only the delta on top of the previous summary (small prompt, ~1s)
        # instead of re-reading the whole transcript (~2-3s).
        summary = ""
        fingerprint = hashlib.sha256(transcript[:200].encode("utf-8")).hexdigest()
        base = _BASE if (
            _BASE.get("fingerprint") == fingerprint
            and transcript.startswith(_BASE.get("transcript", "\x00"))
            and len(transcript) > len(_BASE.get("transcript", ""))
        ) else None
        if base:
            remainder = transcript[len(base["transcript"]):]
            if remainder.strip():
                summary = await _summarize(
                    _INCREMENTAL_USER_TMPL.format(
                        old_summary=base["summary"], new_turns=remainder
                    )
                )
        if not summary:
            summary = await _summarize(transcript)
        if not summary:
            return messages, info
        if len(_SUMMARY_CACHE) >= _CACHE_MAX:
            _SUMMARY_CACHE.pop(next(iter(_SUMMARY_CACHE)))
        _SUMMARY_CACHE[key] = summary
        _BASE.clear()
        _BASE.update({"fingerprint": fingerprint, "transcript": transcript,
                      "summary": summary})
        compacted = [{"role": "system", "content": f"[خلاصهٔ گفتگوی پیشین]\n{summary}"}] + recent
        info.update(compacted=True, old_turns=len(old), cached=False,
                    summary_chars=len(summary))
        log.info(
            "Context compacted: %d old turns → %d-char summary (%s, est %d tokens)",
            len(old), len(summary),
            "incremental" if base else "full", info["estimate_tokens"],
        )
        return compacted, info
    except Exception as e:
        log.warning("Context compaction skipped (LLM error): %s", e)
        return messages, {"compacted": False, "error": str(e)}