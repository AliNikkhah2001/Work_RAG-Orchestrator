"""Rule-gated LLM judge for Dana magic widgets.

Contract
--------
1. ``magic_precheck`` — high-**recall** cheap gates per family. Empty ⇒ skip LLM.
2. When precheck hits (except clarify chips), call the versioned KB prompt
   ``magic_judge`` (or family-specific ``magic_judge_*``) via content_registry.
3. LLM **validates** the candidate; on miss/failure fall back to deterministic
   materialization of the precheck families.

Prompts live in ``packs/prompts/orchestrator/vX.Y.Z.yaml`` (KB registry) and the
orchestrator vendored fallback YAML — never hardcoded in graph nodes.
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
    "باند",
    "نمایش",
)
_PRODUCT = (
    "چک",
    "صیاد",
    "حقیقی",
    "حقوقی",
    "تسهیلات",
    "cheque",
    "person",
    "legal",
    "گزارش چک",
    "اشخاص حقیقی",
    "اشخاص حقوقی",
)
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
    qu = q.upper()
    return (
        ("اعتبارسنجی ایران" in q)
        or ("شرکت اعتبارسنجی" in q)
        or ("امتیازدهی اعتباری ایران" in q)
        or ("ICS" in qu)
        or ("ایران کریدیت" in q)
    )


def _shareholders_gate(query: str) -> bool:
    """High-recall: any shareholder ask that plausibly targets ICS."""
    q = query or ""
    if not any(k in q for k in ("سهامدار", "سهام‌دار", "سهام دار", "ترکیب سهام", "سهام شرکت")):
        return False
    if _ics_company_query(q):
        return True
    # Recall: «سهامداران» + اعتبارسنجی / شرکت without full ICS phrase
    return ("اعتبارسنجی" in q) or ("شرکت" in q and "ایران" in q)


def _creditors_gate(query: str) -> bool:
    """High-recall: data-provider / creditor asks about ICS."""
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
            "منابع اطلاعاتی",
            "دیتا پروایدر",
            "پروایدر",
            "data provider",
            "از کجا داده",
            "داده‌ها را از",
            "داده را از",
            "تأمین اطلاعات",
            "تامین اطلاعات",
        )
    )
    if not hit:
        if ("داده" in q or "دیتا" in q) and any(
            k in q for k in ("بانک", "نهاد", "لیزینگ", "فین‌تک", "بیمه", "نئوبانک", "لندتک")
        ):
            hit = True
    if not hit:
        return False
    if _ics_company_query(q):
        return True
    return ("اعتبارسنجی" in q) or ("شرکت" in q and "ایران" in q)


def _timeline_gate(query: str) -> bool:
    """High-recall: ICS history / intro / founding / products overview."""
    q = query or ""
    # More specific ICS widgets win when both would match — still gate timeline;
    # LLM picks the tighter widget when multiple families are candidated.
    hist = any(
        k in q
        for k in (
            "تاریخچه",
            "تأسیس",
            "تاسیس",
            "خط زمان",
            "تایم‌لاین",
            "تایم لاین",
            "timeline",
            "سابقه",
            "بنیان",
            "شروع به کار",
            "کی تأسیس",
            "کی تاسیس",
            "چه سالی",
            "تاریخ",
            "چه زمانی",
            "از کی",
        )
    )
    intro = any(
        k in q
        for k in (
            "چیست",
            "چی هست",
            "چیه",
            "معرفی",
            "درباره",
            "توضیح",
            "شرح",
            "چه می‌کند",
            "چه میکند",
            "چیکار",
            "چه کار",
            "چه کاری",
            "چه کرد",
            "چه کرده",
            "چه اقدام",
            "چه فعالیت",
            "فعالیت",
            "ماموریت",
            "مأموریت",
            "محصولات",
            "که بود",
            "که هست",
            "کی بود",
            "کی هست",
            "چه شرکتی",
            "چه فرقی",
            "فرقی داره",
            "فرق دارد",
            "تفاوت",
            "متفاوت",
            "بقیه شرکت",
            "سایر شرکت",
        )
    )
    if not (hist or intro):
        return False
    if _ics_company_query(q):
        return True
    return ("شرکت" in q and "اعتبارسنجی" in q) or ("اعتبارسنجی ایران" in q)


def _ambiguous_colour_rank(query: str, product_known: bool = False) -> bool:
    """Colour band ask without چک/حقیقی/حقوقی → clarify path, not score magic.

    *product_known* is set when the product was already resolved (e.g. the user
    just answered a colour clarify with «گزارش چک»), so the rank is no longer
    ambiguous and the gauge may fire instead of asking again.
    """
    if product_known:
        return False
    q = query or ""
    if not _query_has_colour(q):
        return False
    if _query_has_product(q):
        return False
    return any(k in q for k in ("رتبه", "رنگ", "گرید", "امتیاز", "یعنی چه", "یعنی چی"))


def _score_query_for_intent(query: str) -> str:
    """Strip company name «امتیازدهی اعتباری» so it does not look like a score ask."""
    q = query or ""
    q = q.replace("امتیازدهی اعتباری", " ")
    q = q.replace("امتیازدهی", " ")
    return q


def _score_has_viz_anchor(query: str) -> bool:
    """True when the query itself carries a score/colour/grade/ladder/layout signal."""
    q = query or ""
    if any(k in q for k in ("نردبان", "ladder")):
        return True
    if any(
        k in q for k in ("عمودی", "افقی", "horizontal", "vertical", "linear", "خطی")
    ):
        return True
    return (
        _query_has_colour(q)
        or _query_has_grade(q)
        or _query_has_cheque_numeric(q)
        or _query_has_credit_numeric(q)
    )


def _score_gate(query: str, product_known: bool = False) -> bool:
    """High-recall score/bands candidate — query-only; LLM rejects weak hits."""
    q = query or ""
    if _ambiguous_colour_rank(q, product_known=product_known):
        return False
    if any(k in q for k in ("نردبان", "ladder")):
        return True
    q_intent = _score_query_for_intent(q)
    intent = any(k in q_intent for k in _SCORE_INTENT)
    if not intent:
        return False
    layout = any(
        k in q for k in ("عمودی", "افقی", "horizontal", "vertical", "linear", "خطی", "نمایش")
    )
    anchor = _score_has_viz_anchor(q)
    product = _query_has_product(q) or any(
        k in q for k in ("گزارش چک", "تسهیلات", "رتبه رنگی", "گزارش اعتبار")
    )
    # High recall: score-intent + (product OR anchor OR layout)
    if product or anchor or layout:
        return True
    return False


def magic_precheck(
    query: str,
    finish_reason: str | None = None,
    *,
    product_known: bool = False,
) -> frozenset[str]:
    """High-recall candidate families. Empty ⇒ **do not** call the LLM.

    *product_known* says the product/report type was already resolved (e.g. the
    user just answered a colour clarify with «گزارش چک»), which removes the
    cheque-vs-facilities ambiguity that normally forces clarify.
    """
    fr = finish_reason or "stop"
    if fr in ("content_filter",):
        return frozenset()
    if fr == "clarify":
        return frozenset({"clarify"})
    q = query or ""
    if not q.strip():
        return frozenset()
    families: set[str] = set()
    if _score_gate(q, product_known=product_known):
        families.add("score")
    if _shareholders_gate(q):
        families.add("shareholders")
    if _creditors_gate(q):
        families.add("creditors")
    if _timeline_gate(q):
        families.add("timeline")
    return frozenset(families)


_FAMILY_PROMPT = {
    frozenset({"score"}): "magic_judge_score",
    frozenset({"timeline"}): "magic_judge_timeline",
    frozenset({"shareholders"}): "magic_judge_shareholders",
    frozenset({"creditors"}): "magic_judge_creditors",
}

# Subspaces that pin an actual credit product/report type (as opposed to a
# loose topic like «general» or «credit_ladder»).
_PRODUCT_LEAVES = frozenset({"cheque", "facilities_person", "facilities_legal"})


def product_resolved(state: dict) -> bool:
    """True when this turn already resolved the report type (post-clarify answer)."""
    if not state.get("declaration_followup"):
        return False
    sub = str(state.get("applied_subspace") or "").strip().lower()
    return sub in _PRODUCT_LEAVES or "cheque" in sub or "facilities" in sub


def build_magic_judge_prompt(query: str, answer: str, families: frozenset[str]) -> str:
    """Load versioned prompt from KB registry / vendored YAML (never node hardcode)."""
    from .content_registry import require_prompt

    fam_key = ", ".join(sorted(families)) or "NONE"
    name = _FAMILY_PROMPT.get(frozenset(families), "magic_judge")
    try:
        return require_prompt(
            name,
            query=query or "",
            answer=(answer or "")[:1200],
            families=fam_key,
        )
    except Exception:
        # Single-family template missing → router prompt
        return require_prompt(
            "magic_judge",
            query=query or "",
            answer=(answer or "")[:1200],
            families=fam_key,
        )


def parse_magic_judge_response(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
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
    """LLM validation — only after ``magic_precheck`` is non-empty (non-clarify)."""
    if not _MAGIC_JUDGE_ENABLED or not families or families == frozenset({"clarify"}):
        return None
    try:
        prompt = build_magic_judge_prompt(query, answer, families)
    except Exception as e:
        log.warning("magic_judge prompt missing from registry: %s", e)
        return None
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
