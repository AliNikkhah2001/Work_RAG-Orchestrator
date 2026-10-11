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
