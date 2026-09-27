"""Verify FAQ direct-answer candidates via LLM call to Guardrails (Gemma).

Checks whether a canned FAQ answer actually addresses the user's question.
Used by retrieve.py to gate FAQ short-circuit: only verified answers get
injected as context; unverified ones fall through to normal retrieval.
"""

from __future__ import annotations

import logging

from ..config import get_settings

log = logging.getLogger(__name__)


async def verify_faq_answer(query: str, faq_answer: str) -> bool:
    """Ask Gemma whether faq_answer actually addresses query.

    Returns True if relevant, False otherwise.
    On any error, returns False (fallback to retrieval — never short-circuit on error).
    """
    settings = get_settings()
    try:
        from .guardrails import _get_shared_client
        client = _get_shared_client(15.0)
        verify_payload = {
            "model": settings.upstream_llm_model,
            "messages": [
                {"role": "system", "content": "آیا پاسخ زیر به سؤال کاربر پاسخ می‌دهد؟ فقط با y یا n جواب بده."},
                {"role": "user", "content": f"سؤال: {query}\nپاسخ کاندید: {faq_answer[:200]}\nاین پاسخ به سؤال جواب می‌دهد؟ (y/n)"},
            ],
            "temperature": 0,
            "max_tokens": 5,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = await client.post(settings.guardrails_chat_url, json=verify_payload, timeout=15)
        data = resp.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip().lower()
        return content.startswith("y")
    except Exception as e:
        log.warning("FAQ verifier error: %s — falling back to retrieval", e)
        return False
