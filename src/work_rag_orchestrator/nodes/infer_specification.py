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
    "enterprise": ["شرکت", "حقوقی", "سهامدار", "شرکتها"],
}

JUDGE_PROMPT = '''شما یک دسته‌بند پیام کاربر هستید. پیام زیر را بررسی کنید:
  پیام: "{msg}"
  گزینه‌های موجود: [تسهیلات افراد, گزارش چک, تسهیلات شرکت‌ها]
  آیا این پیام یک "تعیین شاخه" است (کاربر مشخص می‌کند از کدام شاخه می‌خواهد) یا یک "سوال" است؟
  - تعیین شاخه: پیام کوتاه (1-4 کلمه) که نام شاخه را می‌برد، مثل "چک" یا "منظورم چک بود" یا "شرکت"
  - سوال: پیام پرسشی که جواب می‌خواهد، مثل "امتیاز چک چیست؟" یا "بدهکار شدم چیکار کنم"
  فقط با یک کلمه پاسخ دهید: "specification" یا "question"'''


def _is_short_with_keyword(msg: str) -> bool:
    """Heuristic fallback: True if msg is 1-4 words and contains a subspace keyword."""
    if not msg or not msg.strip():
        return False
    # word_tokenize via whitespace / non-space sequence (uses re per spec imports)
    tokens = re.findall(r"\S+", msg.strip())
    if len(tokens) > 4:
        return False
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
