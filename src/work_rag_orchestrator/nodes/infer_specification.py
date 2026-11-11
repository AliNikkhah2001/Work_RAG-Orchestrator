"""LLM judge for specification vs question. is_specification(msg) -> bool, sticky subspace until re-specification."""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..clients.guardrails import GuardrailsClient
from ..schemas import GuardrailsChatRequest
from ..config import get_settings
from ..state import RAGState

log = logging.getLogger(__name__)

SUBSPACE_KEYWORDS = {
    "cheque": ["چک", "صیاد", "پیچک", "برگشت", "نقدشده"],
    "facilities": ["تسهیلات", "وام", "قسط", "بازپرداخت"],
    "enterprise": ["حقوقی", "سهامدار", "شرکتها"],
    "credit_ladder": ["نردبان", "نردبان اعتباری"],
    "guarantor": ["کفیل", "کفالت", "وجه‌الکفاله"],
    "supplier_legal": ["تامین کننده", "تامین‌کننده"],
    "customers": ["مشتریان من", "پنل کسب"],
    "general": ["عمومی", "عام", "معرفی", "مدیر", "هیات مدیره", "هیئت مدیره", "کاربرد", "کاربردها", "شرکت"],
}

# T5 scattered clarify gate: vague phrases without specific subspace keywords => ambiguous
SCATTERED_VAGUE_PHRASES = ["وضعیت اعتباری", "اعتبارسنجی", "بدهکار", "بدهی"]
# Strict subspace keywords that disambiguate a query (if any present, not scattered)
SCATTERED_SUBSPACE_KWS = [
    "چک", "صیاد", "پیچک", "برگشت",
    "تسهیلات", "وام", "قسط", "بازپرداخت",
    "حقوقی", "حقیقی", "سهامدار",
    "شرکت", "معرفی", "مدیر", "هیات مدیره", "هیئت مدیره",
    "عمومی", "عام", "کاربرد",
    "نردبان", "کفیل", "کفالت", "تامین", "مشتریان من", "پنل",
]
# Generic credit words that together with short length signal scattered even if not exact vague phrase
SCATTERED_GENERIC_WORDS = ["وضعیت", "اعتبار", "اعتباری", "امتیاز", "بدهکار", "بدهی"]


def is_scattered_ambiguous(query: str) -> bool:
    """Detect ambiguous/scattered queries that must trigger clarify.

    Returns True if query contains vague terms (وضعیت اعتباری / اعتبارسنجی / بدهکار)
    without any specific subspace keyword (چک, تسهیلات, اشخاص حقیقی/حقوقی, شرکت, etc.).
    Also handles short generic queries (<=6 words) containing generic credit words
    without disambiguating keywords.
    """
    q = (query or "").replace("‌", " ").strip()
    if not q:
        return False
    has_vague = any(v in q for v in SCATTERED_VAGUE_PHRASES)
    has_subspace = any(kw in q for kw in SCATTERED_SUBSPACE_KWS)
    if has_vague and not has_subspace:
        return True
    # Short generic fallback: e.g. "وضعیت اعتباری چطور است" already caught, but also
    # covers edge like "اعتبار چطوره" if ever probed
    tokens = q.split()
    if len(tokens) <= 6 and not has_subspace:
        # Exclude known non-scattered entities that look like general but are specific
        if "قاسم" in q or "بور" in q:
            return False
        if any(w in q for w in SCATTERED_GENERIC_WORDS):
            return True
    return False

def _is_short_with_keyword(msg: str) -> bool:
    """Heuristic fallback: True if msg is 1-6 words and contains a subspace keyword.

    Extended: phrase "اشخاص حقیقی"/"اشخاص حقوقی" + "تسهیلات"/"اعتبارسنجی"
    is specification regardless of length, provided it is not interrogative.
    Also handles "هیچ کدوم مدیر شرکته" → general specification.
    """
    if not msg or not msg.strip():
        return False
    # Long-phrase exception: explicit subspace declaration with اشخاص حقیقی/حقوقی
    # e.g. "اعتبارسنجی تسهیلات برای اشخاص حقیقی" (5 words) → specification
    # regardless of token count, provided it is not interrogative.
    if "اشخاص حقیقی" in msg or "اشخاص حقوقی" in msg:
        if "؟" in msg or "?" in msg:
            return False
        for w in ("چیست", "چطور", "چگونه", "آیا", "چرا", "کجاست", "چیه", "چيست", "کیه", "کيست"):
            if w in msg:
                return False
        return True
    # Interrogative gate — even a short keyword phrase is a question if it
    # contains ؟/? or any interrogative word. This prevents
    # "چک برگشتی چیست؟" (3 words, contains چک) from being misclassified.
    if "؟" in msg or "?" in msg:
        return False
    for w in ("چیست", "چطور", "چگونه", "آیا", "چرا", "کجاست", "چیه", "چيست", "کیه", "کيست"):
        if w in msg:
            return False
    tokens = re.findall(r"\S+", msg.strip())
    if len(tokens) > 6:
        return False
    # "هیچ کدوم" + manager/company → general specification even if combined tokens 3-6
    if ("هیچ کدوم" in msg or "هیچکدوم" in msg) and any(kw in msg for kw in ["مدیر","هیات مدیره","هیئت مدیره","معرفی","عمومی","شرکت","عام"]):
        return True
    # HAS_KW: any keyword from SUBSPACE_KEYWORDS appears as substring
    for kws in SUBSPACE_KEYWORDS.values():
        for kw in kws:
            if kw in msg:
                return True
    return False


async def is_specification(msg: str, request_id: str) -> bool:
    """LLM judge: True if msg is a subspace specification, False if question."""
    # Hard gate: interrogative is always a question, never a specification
    # Prevents "وضعیت اعتباری چطور است" from being misclassified via LLM hallucination
    if "؟" in msg or "?" in msg:
        return False
    for w in ("چیست", "چطور", "چگونه", "آیا", "چرا", "کجاست", "چیه", "چيست", "کیه", "کيست"):
        if w in msg:
            return False
    try:
        from ..content_registry import require_prompt

        settings = get_settings()
        prompt = require_prompt("specify_judge", msg=msg)
        request = GuardrailsChatRequest(
            model=settings.upstream_llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=16,
        )
        async with GuardrailsClient() as client:
            response = await client.chat_completion(request, request_id)
        content = ""
        if response.choices:
            content = response.choices[0].message.get("content", "") or ""
        lower = content.lower().strip()
        if "specification" in lower:
            return True
        if "question" in lower:
            return False
        return _is_short_with_keyword(msg)
    except Exception:
        log.exception("is_specification LLM judge failed, fallback to heuristic for request %s", request_id)
        return _is_short_with_keyword(msg)


SUBSPACE_LABELS = {
    "cheque": "گزارش اعتباری چک",
    "facilities_person": "اعتبارسنجی تسهیلات برای اشخاص حقیقی (افراد)",
    "facilities_legal": "اعتبارسنجی تسهیلات برای اشخاص حقوقی (شرکت‌ها)",
    "credit_ladder": "نردبان اعتباری",
    "guarantor": "اعتبارسنجی کفیل",
    "supplier_legal": "تامین‌کننده اطلاعات",
    "customers_person": "مشتریان من (حقیقی)",
    "customers_legal": "مشتریان من (حقوقی)",
    "general": "پایگاه دانش عمومی (معرفی، هیات مدیره، کاربردها)",
}

async def infer_subspace_llm(query: str, history: str, request_id: str) -> str | None:
    try:
        from ..content_registry import require_prompt
        from ..subspace_routing import is_cross_product_impact, parse_subspace_label

        if is_cross_product_impact(query):
            return None
        settings = get_settings()
        prompt = require_prompt(
            "suggest_subspace",
            history=history[:1000],
            query=query[:500],
        )
        async with GuardrailsClient() as client:
            req = GuardrailsChatRequest(model=settings.upstream_llm_model, messages=[{"role": "user", "content": prompt}], max_tokens=16, temperature=0.0)
            resp = await client.chat_completion(req, request_id)
            ans = resp.choices[0].message.get("content","").strip().lower() if resp.choices else ""
            return parse_subspace_label(ans)
    except Exception: return None
