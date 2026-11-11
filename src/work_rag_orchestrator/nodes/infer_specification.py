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


def is_definition_question(query: str) -> bool:
    """True for short چیست/چیه definition asks (FAQ-eligible, not scattered credit-status).

    «اعتبارسنجی چیست» should take the FAQ/general path; «وضعیت اعتباری چطور است»
    uses چطور and remains scattered.
    """
    q = (query or "").replace("‌", " ").strip()
    if not q:
        return False
    # Only definition particles — not چطور/چگونه (those stay scattered for credit-status).
    if not any(w in q for w in ("چیست", "چيه", "چیه", "چيست")):
        return False
    tokens = q.replace("؟", " ").replace("?", " ").split()
    return len(tokens) <= 6


# Back-compat alias for early call sites
_is_definition_question = is_definition_question


def is_scattered_ambiguous(query: str) -> bool:
    """Detect ambiguous/scattered queries that must trigger clarify.

    Returns True if query contains vague terms (وضعیت اعتباری / اعتبارسنجی / بدهکار)
    without any specific subspace keyword (چک, تسهیلات, اشخاص حقیقی/حقوقی, شرکت, etc.).
    Also handles short generic queries (<=6 words) containing generic credit words
    without disambiguating keywords.

    Definition asks (چیست/چیه) are exempt so FAQ/direct-answer can apply
    (e.g. «اعتبارسنجی چیست» → general/FAQ, not nardban/kafil clarify).
    """
    q = (query or "").replace("‌", " ").strip()
    if not q:
        return False
    # P2.2: definition form of vague terms is FAQ-eligible, not scattered clarify
    if is_definition_question(q):
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


def label_subspace_from_query(query: str) -> Optional[str]:
    """Map an explicit non-scattered query to a canonical L2 subspace label.

    Used by retrieve/api when sticky is empty but the user clearly named a
    leaf (e.g. «اعتبارسنجی تسهیلات برای اشخاص حقیقی» → facilities_person).
    Returns None when the query does not carry a decisive subspace keyword.
    """
    text = (query or "").strip()
    if not text:
        return None
    low = text.lower()
    has_facility = any(kw in text for kw in ("تسهیلات", "وام", "قسط"))
    has_cheque = "چک" in text
    if ("هیچ کدوم" in text or "هیچکدوم" in text) and any(
        kw in text for kw in ("مدیر", "هیات مدیره", "هیئت مدیره", "معرفی", "عمومی", "شرکت", "عام")
    ):
        return "general"
    if "نردبان" in text:
        return "credit_ladder"
    if any(kw in text for kw in ("کفیل", "کفالت")):
        return "guarantor"
    if "تامین" in text:
        return "supplier_legal"
    if "مشتریان من" in text or text.strip() == "پنل":
        return "customers_legal" if ("حقوقی" in text or "شرکت" in text) else "customers_person"
    # Facilities before bare cheque: «تسهیلات … اشخاص حقیقی» must not lose to
    # incidental چک wording elsewhere in the thread/answer.
    if has_facility and ("اشخاص حقیقی" in text or ("حقیقی" in text and "حقوقی" not in text)):
        return "facilities_person"
    if has_facility and ("اشخاص حقوقی" in text or "حقوقی" in text or "شرکت" in text):
        return "facilities_legal"
    if has_cheque and not has_facility:
        return "cheque"
    if has_cheque and "گزارش" in text and "اعتبار" in text:
        return "cheque"
    if any(kw in text for kw in ("مدیر", "هیات مدیره", "هیئت مدیره", "معرفی", "عمومی", "عام", "کاربرد")) or "general" in low:
        return "general"
    if "شرکت" in text and not has_facility and not has_cheque:
        return "general"
    if has_facility:
        return "facilities_person"
    return None


async def is_specification(msg: str, request_id: str) -> bool:
    """LLM judge: True if msg is a subspace specification, False if question."""
    # Hard gate: interrogative is always a question, never a specification
    # Prevents "وضعیت اعتباری چطور است" from being misclassified via LLM hallucination
    if "؟" in msg or "?" in msg:
        return False
    for w in ("چیست", "چطور", "چگونه", "آیا", "چرا", "کجاست", "چیه", "چيست", "کیه", "کيست"):
        if w in msg:
            return False
    # Hard gate: explicit اشخاص حقیقی/حقوقی *declarations* are specifications even
    # when the LLM later labels them as topical questions (conv_facilities_person_02 T3).
    # Imperative answer-seeking («… بگو») stays a question so retrieve still answers.
    _answer_seek = any(w in msg for w in ("بگو", "بگویید", "توضیح", "شرح بده", "بفرمایید", "بفرما"))
    if (
        not _answer_seek
        and ("اشخاص حقیقی" in msg or "اشخاص حقوقی" in msg)
        and _is_short_with_keyword(msg)
    ):
        return True
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
