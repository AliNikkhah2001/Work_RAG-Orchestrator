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

JUDGE_PROMPT = '''شما یک دسته‌بند پیام کاربر هستید. پیام زیر را بررسی کنید:
  پیام: "{msg}"
  گزینه‌های موجود: [تسهیلات افراد, گزارش چک, تسهیلات شرکت‌ها, عمومی (معرفی، هیات مدیره، کاربردها)]
  آیا این پیام یک "تعیین شاخه" است (کاربر مشخص می‌کند از کدام شاخه می‌خواهد) یا یک "سوال" است؟
  - تعیین شاخه: ONLY پیام کوتاه حداکثر 6 کلمه که فقط نام/کلیدواژه شاخه را می‌برد بدون کلمه پرسشی و بدون علامت سوال، مثل "چک" (1 کلمه، بدون ؟، فقط کلیدواژه) یا "منظورم چک بود" یا "مدیر شرکته" یا "هیچ کدوم مدیر شرکته" یا "عمومی". اگر پیام بیشتر از 6 کلمه باشد یا پرسشی باشد، specification نیست.
    استثنا: اگر پیام حاوی عبارت دقیق "اشخاص حقیقی" یا "اشخاص حقوقی" همراه با "تسهیلات" یا "اعتبارسنجی" باشد و پرسشی نباشد (بدون "؟" و بدون کلمات پرسشی "چیست/چطور/چگونه/آیا/چرا/کجاست/چیه")، حتی اگر بیش از 6 کلمه باشد specification است — مثال "اعتبارسنجی تسهیلات برای اشخاص حقیقی" → specification.
    نکته: "مدیر"، "هیات مدیره"، "هیئت مدیره"، "معرفی شرکت"، "عمومی"، "هیچ کدوم مدیر شرکته" → عمومی specification است. "شرکت" تنها بدون تسهیلات/چک → عمومی؛ "شرکت" + تسهیلات/حقوقی → تسهیلات شرکت‌ها.
  - سوال: هر پیام پرسشی که جواب می‌خواهد — حتی اگر کلمه "چک" داشته باشد، باز هم سوال است. اگر پیام حاوی هر یک از کلمات پرسشی "چیست"، "چطور"، "چگونه"، "آیا"، "چرا"، "کجاست"، "کیه" یا علامت پایانی "؟" باشد، حتما "question" است، صرف‌نظر از وجود "چک". مثال: "چک" → specification اما "چک برگشتی چیست؟" → question و "چک برگشتی چیست و چگونه رفع اثر می‌شود؟" → question
  فقط با یک کلمه پاسخ دهید: "specification" یا "question"'''


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
        settings = get_settings()
        prompt = JUDGE_PROMPT.format(msg=msg)
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
    "general": "پایگاه دانش عمومی (معرفی، هیات مدیره، کاربردها)",
}

SUGGEST_PROMPT = """شما دسته‌بند سوال کاربر هستید. تاریخچه:
{history}
سوال فعلی بدون شاخه مشخص: "{query}"
گزینه‌ها: [تسهیلات افراد, گزارش چک, تسهیلات شرکت‌ها, عمومی]
- اگر سوال درباره چک/صیاد/برگشت است → گزارش چک
- اگر درباره وام/تسهیلات/قسط/بازپرداخت برای افراد است → تسهیلات افراد
- اگر درباره شرکت/حقوقی/سهامدار است → تسهیلات شرکت‌ها
- اگر درباره معرفی/هیات مدیره/کاربردها است → عمومی
- اگر سوال هم‌زمان حاوی «چک» و یکی از «تاثیر/تأثیر/اعتبارسنجی/تسهیلات/امتیاز» است، بین «گزارش چک» (سابقه صرف) و «اعتبارسنجی تسهیلات» مردد است → unclear
- اگر مبهم و قابل انتساب به چند شاخه است → unclear
- مثال: «من چکم برگشت خورده چه تاثیری داره» → unclear (چک + تاثیر/اعتبارسنجی/تسهیلات/امتیاز هم‌زمان)
فقط یک کلمه: cheque / facilities_person / facilities_legal / general / unclear
 """

async def infer_subspace_llm(query: str, history: str, request_id: str) -> str | None:
    try:
        settings = get_settings()
        prompt = SUGGEST_PROMPT.format(history=history[:1000], query=query[:500])
        async with GuardrailsClient() as client:
            req = GuardrailsChatRequest(model=settings.upstream_llm_model, messages=[{"role": "user", "content": prompt}], max_tokens=16, temperature=0.0)
            resp = await client.chat_completion(req, request_id)
            ans = resp.choices[0].message.get("content","").strip().lower() if resp.choices else ""
            for k in ["cheque","facilities_person","facilities_legal","general"]:
                if k in ans: return k
            if "unclear" in ans: return None
            return None
    except Exception: return None
