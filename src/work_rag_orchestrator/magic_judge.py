"""Rule-gated LLM judge for Dana magic widgets.

Latency contract
----------------
* ``magic_precheck(query, finish_reason)`` is cheap and **must** return empty
  for the vast majority of turns (generic FAQ, greeting, law articles, provider
  counts, company prose without ICS-widget intent).
* The LLM is called **only** when precheck returns a non-empty family set.
* Clarify cards stay deterministic (no LLM) when ``finish_reason=clarify``.

The judge never invents a score from incidental answer digits (ماده ۵، سال ۱۳۸۵،
۶۸ تأمین‌کننده). Scores / colours must be anchored in the **user query**.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

_MAGIC_JUDGE_ENABLED = os.getenv("MAGIC_LLM_JUDGE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
_MAGIC_JUDGE_TIMEOUT = float(os.getenv("MAGIC_JUDGE_TIMEOUT_S", "8"))
_MAGIC_JUDGE_MAX_TOKENS = int(os.getenv("MAGIC_JUDGE_MAX_TOKENS", "180"))

_SCORE_INTENT = (
    "امتیاز",
    "رتبه",
    "رنگ",
    "نردبان",
    "گرید",
    "نمره",
    "ladder",
    "عمودی",
    "افقی",
    "horizontal",
    "vertical",
    "linear",
    "خطی",
)
_PRODUCT = ("چک", "صیاد", "حقیقی", "حقوقی", "تسهیلات", "cheque", "person", "legal")
_COLOURS = ("آبی", "ابي", "سبز", "زرد", "نارنجی", "نارنجي", "قرمز")


def _fa_lat(text: str) -> str:
    return (text or "").translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))


def _mask_non_score_numbers(text: str) -> str:
    t = _fa_lat(text)
    t = re.sub(r"ماده\s*\(?\s*\d{1,3}\s*\)?", " ", t)
    t = re.sub(r"\b(13|14|19|20)\d{2}\b", " ", t)
    t = re.sub(r"سال\s+\d{2,4}", " ", t)
    return t


def _query_has_cheque_numeric(q: str) -> bool:
    t = _mask_non_score_numbers(q)
    for m in re.finditer(r"(?<![.\d])(\d{1,3})(?![.\d])", t):
        n = int(m.group(1))
        if 0 <= n <= 100:
            return True
    return False


def _query_has_credit_numeric(q: str) -> bool:
    t = _mask_non_score_numbers(q)
    for m in re.finditer(r"(?<![.\d])(\d{3})(?![.\d])", t):
        n = int(m.group(1))
        if 250 <= n <= 900:
            return True
    return False


def _query_has_grade(q: str) -> bool:
    return bool(re.search(r"\b[A-Ea-e]\s*[1-3]\b", _fa_lat(q)))


def _query_has_colour(q: str) -> bool:
    return any(c in (q or "") for c in _COLOURS)


def _query_has_product(q: str) -> bool:
    ql = (q or "").lower()
    return any(p.lower() in ql if p.isascii() else p in (q or "") for p in _PRODUCT)


def _ics_company_query(query: str) -> bool:
    q = query or ""
    return ("اعتبارسنجی ایران" in q) or ("شرکت اعتبارسنجی" in q) or ("ICS" in q.upper()) or (
        "امتیازدهی اعتباری ایران" in q
    )


def _shareholders_gate(query: str) -> bool:
    q = query or ""
    if not any(k in q for k in ("سهامدار", "سهام‌دار", "سهام دار")):
        return False
    return _ics_company_query(q)


def _creditors_gate(query: str) -> bool:
    q = query or ""
    hit = any(
        k in q
        for k in (
            "تامین‌کننده",
            "تأمین‌کننده",
            "تامین کننده",
            "تأمین کننده",
            "تامین‌کنندگان",
            "تأمین‌کنندگان",
            "تامین کنندگان",
            "تأمین کنندگان",
            "منابع داده",
            "دیتا پروایدر",
            "data provider",
        )
    )
    if not hit:
        if ("داده" in q or "دیتا" in q) and any(
            k in q for k in ("بانک", "نهاد", "لیزینگ", "فین‌تک", "بیمه")
        ):
            hit = True
    if not hit:
        return False
    return _ics_company_query(q)


def _timeline_gate(query: str) -> bool:
    q = query or ""
    # Shareholders / creditors asks are more specific than company intro
    if _shareholders_gate(q) or _creditors_gate(q):
        return False
    if any(
        k in q
        for k in (
            "تاریخچه",
            "تأسیس",
            "تاسیس",
            "خط زمان",
            "تایم‌لاین",
            "تایم لاین",
            "timeline",
            "سابقه شرکت اعتبارسنجی",
        )
    ):
        return _ics_company_query(q) or ("شرکت" in q and "اعتبارسنجی" in q)
    if _ics_company_query(q) and any(
        k in q for k in ("چیست", "چی هست", "معرفی", "درباره", "توضیح", "چه می‌کند", "فعالیت", "ماموریت", "محصولات")
    ):
        return True
    return False


def _ambiguous_colour_rank(query: str) -> bool:
    """Colour band ask without چک/حقیقی/حقوقی → must clarify, not gauge."""
    q = query or ""
    if not _query_has_colour(q):
        return False
    if _query_has_product(q):
        return False
    return any(k in q for k in ("رتبه", "رنگ", "گرید", "امتیاز"))


def _score_gate(query: str) -> bool:
    """Score/bands widget candidate — query-only, never answer prose."""
    q = query or ""
    if _ambiguous_colour_rank(q):
        return False
    if any(k in q for k in ("نردبان", "ladder")):
        return True
    layout = any(
        k in q for k in ("عمودی", "افقی", "horizontal", "vertical", "linear", "خطی", "نمایش")
    )
    intent = any(k in q for k in _SCORE_INTENT)
    if not intent:
        return False
    anchor = (
        _query_has_colour(q)
        or _query_has_grade(q)
        or _query_has_cheque_numeric(q)
        or _query_has_credit_numeric(q)
    )
    product = _query_has_product(q) or any(k in q for k in ("گزارش چک", "تسهیلات", "اعتباری"))
    # Explicit layout + score intent (e.g. «امتیاز … عمودی») with product/scope
    if layout and product:
        return True
    if not anchor:
        return False
    # Require product / report scope so «امتیاز من چند است؟» stays dark
    return product


def magic_precheck(query: str, finish_reason: str | None = None) -> frozenset[str]:
    """Return candidate magic families. Empty ⇒ **do not** call the LLM."""
    fr = finish_reason or "stop"
    if fr in ("content_filter",):
        return frozenset()
    if fr == "clarify":
        return frozenset({"clarify"})
    q = query or ""
    if not q.strip():
        return frozenset()
    families: set[str] = set()
    if _score_gate(q):
        families.add("score")
    if _shareholders_gate(q):
        families.add("shareholders")
    if _creditors_gate(q):
        families.add("creditors")
    if _timeline_gate(q):
        families.add("timeline")
    return frozenset(families)


_JUDGE_PROMPT = """نقش: مسیریاب ویجت Dana برای سامانه اعتبارسنجی ایران (ICS).
فقط وقتی یکی از خانواده‌های کاندید زیر مجاز است تصمیم بگیر؛ در غیر این صورت fire=false.

کاندیدها: {families}

ویجت‌ها:
- NONE: هیچ ویجتی
- CREDIT_SCORE_GAUGE: فقط اگر کاربر صریحاً درباره امتیاز/رتبه/رنگ/گرید بپرسد و در خودِ پرسش عدد امتیاز یا رنگ یا کد A1–E3 باشد. هرگز از سال، ماده قانون، تعداد تأمین‌کننده، یا لیست محصولات امتیاز نساز.
- CREDIT_SCORE_BANDS: نردبان/ladder
- COMPANY_TIMELINE: تاریخچه/تأسیس/معرفی شرکت اعتبارسنجی ایران
- SHAREHOLDERS: سهامداران شرکت اعتبارسنجی ایران
- CREDITORS: تأمین‌کنندگان داده شرکت اعتبارسنجی ایران

مدل امتیاز: cheque (۱–۱۰۰، چک) | person (۲۵۰–۹۰۰، حقیقی) | legal (۲۵۰–۹۰۰، حقوقی)
variant پیش‌فرض ring؛ در صورت نردبان vertical/horizontal؛ خطی=linear؛ عمودی=vertical؛ افقی=horizontal.

پرسش کاربر:
{query}

پاسخ مدل (فقط برای زمینه؛ امتیاز را از اینجا اختراع نکن):
{answer}

خروجی فقط JSON یک خطی، یکی از:
{{"fire":false}}
{{"fire":true,"name":"CREDIT_SCORE_GAUGE","params":{{"model":"cheque","score":80,"band":null,"variant":"ring"}}}}
{{"fire":true,"name":"CREDIT_SCORE_BANDS","params":{{"model":"person","variant":"vertical"}}}}
{{"fire":true,"name":"COMPANY_TIMELINE","params":{{"variant":"vertical"}}}}
{{"fire":true,"name":"SHAREHOLDERS","params":{{}}}}
{{"fire":true,"name":"CREDITORS","params":{{}}}}
"""


def build_magic_judge_prompt(query: str, answer: str, families: frozenset[str]) -> str:
    try:
        from .content_registry import require_prompt

        return require_prompt(
            "magic_judge",
            query=query or "",
            answer=(answer or "")[:1200],
            families=", ".join(sorted(families)) or "NONE",
        )
    except Exception:
        return _JUDGE_PROMPT.format(
            families=", ".join(sorted(families)) or "NONE",
            query=query or "",
            answer=(answer or "")[:1200],
        )


def parse_magic_judge_response(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    # Strip fences / thinking leftovers
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if not obj.get("fire"):
        return {"fire": False}
    name = str(obj.get("name") or "").strip().upper()
    allowed = {
        "CREDIT_SCORE_GAUGE",
        "CREDIT_SCORE_BANDS",
        "COMPANY_TIMELINE",
        "SHAREHOLDERS",
        "CREDITORS",
    }
    if name not in allowed:
        return {"fire": False}
    params = obj.get("params") if isinstance(obj.get("params"), dict) else {}
    return {"fire": True, "name": name, "params": params}


async def call_magic_judge(
    query: str,
    answer: str,
    families: frozenset[str],
    *,
    request_id: str = "",
) -> dict[str, Any] | None:
    """LLM call — only invoke after ``magic_precheck`` is non-empty."""
    if not _MAGIC_JUDGE_ENABLED or not families or families == frozenset({"clarify"}):
        return None
    prompt = build_magic_judge_prompt(query, answer, families)
    try:
        from .clients.guardrails import GuardrailsClient
        from .schemas import GuardrailsChatRequest
        from .config import get_settings

        settings = get_settings()
        async with GuardrailsClient(timeout=_MAGIC_JUDGE_TIMEOUT) as client:
            req = GuardrailsChatRequest(
                model=settings.upstream_llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_MAGIC_JUDGE_MAX_TOKENS,
                temperature=0.0,
            )
            resp = await client.chat_completion(req, request_id or "magic-judge")
            text = ""
            if resp.choices:
                text = (resp.choices[0].message.get("content") or "").strip()
            return parse_magic_judge_response(text)
    except Exception as e:
        log.warning("magic_judge LLM failed (falling back to rules): %s", e)
        return None
