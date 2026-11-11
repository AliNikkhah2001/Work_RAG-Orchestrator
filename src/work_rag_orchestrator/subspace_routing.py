"""Shared L2 subspace routing helpers (taxonomy-aligned product folders).

Canonical ids match KB ``packs/taxonomy`` nodes. Embedding prototypes may still
be 4-way until rebuilt; routing maps / sticky / clarify use the full L2 set.
``subspace_ambiguity_route`` stays off until 9-way protos exist.
"""
# CI stage apply: L2 sticky subspaces (jenkins rebuild marker)


from __future__ import annotations

# Full L2 set used for sticky / clarify / suggest parsing.
L2_SUBSPACES: tuple[str, ...] = (
    "cheque",
    "facilities_person",
    "facilities_legal",
    "credit_ladder",
    "guarantor",
    "supplier_legal",
    "customers_person",
    "customers_legal",
    "general",
)

# Labels returned when Gemma/heuristics cannot commit (not sticky ids).
NON_STICKY_LABELS = frozenset({"ambiguous", "unclear"})

DEFAULT_SUBSPACE_MAP: dict[str, str] = {
    "cheque": "گزارش اعتبارسنجی چک",
    "facilities_person": "اشخاص حقیقی/اعتبارسنجی تسهیلات",
    "facilities_legal": "اشخاص حقوقی/اعتبارسنجی تسهیلات",
    "credit_ladder": "اشخاص حقیقی/نردبان اعتباری",
    "guarantor": "اشخاص حقیقی/کفیل",
    "supplier_legal": "اشخاص حقوقی/عضویت در پایگاه داده اعتباری به عنوان تامین کننده اطلاعات",
    "customers_person": "اشخاص حقیقی/گزارش اعتبارسنجی مشتریان من (مرتبط با کسب‌وکارها)",
    "customers_legal": "اشخاص حقوقی/گزارش اعتبارسنجی مشتریان من",
    "general": "پایگاه دانش عمومی",
}

DEFAULT_KEYWORD_BOOST: dict[str, float] = {
    "cheque": 3.5,
    "facilities_person": 3.0,
    "facilities_legal": 3.0,
    "credit_ladder": 3.0,
    "guarantor": 3.0,
    "supplier_legal": 3.0,
    "customers_person": 3.0,
    "customers_legal": 3.0,
    "general": 2.5,
}

SUBSPACE_SHIFT_KEYWORDS: dict[str, list[str]] = {
    "cheque": ["چک", "برگشتی", "صك", "صیاد", "پیچک"],
    "facilities_person": ["تسهیلات", "وام", "قسط", "حقیقی", "شخصی"],
    "facilities_legal": ["حقوقی", "سهامدار", "تسهیلات حقوقی"],
    "credit_ladder": ["نردبان", "نردبان اعتباری"],
    "guarantor": ["کفیل", "کفالت", "وجه‌الکفاله", "وجه الکفاله"],
    "supplier_legal": ["تامین کننده", "تامین‌کننده", "تامین کننده اطلاعات"],
    "customers_person": ["مشتریان من", "پنل کسب و کار", "پنل کسب‌وکار"],
    "customers_legal": ["مشتریان من حقوقی", "پنل حقوقی"],
    "general": [
        "مدیر",
        "هیات مدیره",
        "هیئت مدیره",
        "قاسم",
        "بور",
        "شخصیت",
        "مشهور",
        "عمومی",
        "عام",
        "سوالات عمومی",
        "سوال عمومی",
        "معرفی شرکت",
        "معرفی",
    ],
}

DECLARATION_ALIASES: dict[str, str] = {
    "شخصی": "facilities_person",
    "حقیقی": "facilities_person",
    "شرکتی": "facilities_legal",
    "حقوقی": "facilities_legal",
    "چک": "cheque",
    "cheque": "cheque",
    "عمومی": "general",
    "general": "general",
    "نردبان": "credit_ladder",
    "نردبان اعتباری": "credit_ladder",
    "کفیل": "guarantor",
    "تامین کننده": "supplier_legal",
    "تامین‌کننده": "supplier_legal",
    "تامین کننده اطلاعات": "supplier_legal",
    "مشتریان من": "customers_person",
    "پنل": "customers_person",
}

# Cheque-report UI color ranks (سبز/آبی/…). Colloquial «ابی» without hamza included.
CHEQUE_COLOR_TOKENS: tuple[str, ...] = (
    "آبی",
    "ابی",
    "سبز",
    "زرد",
    "قرمز",
    "نارنجی",
    "بنفش",
    "سفید",
    "مشکی",
)


def query_has_cheque_color_rank(query: str) -> bool:
    """True when the user refers to colour bands used on the cheque score report."""
    q = (query or "").replace("ي", "ی").replace("ك", "ک")
    return any(tok in q for tok in CHEQUE_COLOR_TOKENS)


def normalize_cheque_color_typos(query: str) -> str:
    """Map common typos so BM25/dense can hit cheque colour FAQs (ابی → آبی)."""
    q = query or ""
    # Whole-word-ish: ابی as colour (not substrings inside other tokens)
    if "آبی" not in q and "ابی" in q:
        q = q.replace("ابی", "آبی")
    return q


def prefer_cheque_for_color_query(query: str, subspace: str | None) -> str | None:
    """Colour-rank questions are cheque-report content; override facilities leaf picks."""
    if not query_has_cheque_color_rank(query):
        return subspace
    if subspace in (None, "facilities_person", "facilities_legal", "general"):
        return "cheque"
    return subspace


def extract_subspace_from_hierarchy(hier: list[str] | str | None) -> str | None:
    """Map folder_hierarchy to an L2 id. Returns None when unknown (no silent person default)."""
    s = "/".join(hier) if isinstance(hier, list) else str(hier or "")
    if not s.strip():
        return None
    if "نردبان" in s:
        return "credit_ladder"
    if "کفیل" in s:
        return "guarantor"
    if "تامین" in s:
        return "supplier_legal"
    if "مشتریان" in s or "پنل" in s:
        return "customers_legal" if "حقوقی" in s else "customers_person"
    if "چک" in s:
        return "cheque"
    if "پایگاه دانش عمومی" in s or "/عمومی" in s or s.endswith("عمومی"):
        return "general"
    if "اشخاص حقوقی" in s and "تسهیلات" in s:
        return "facilities_legal"
    if "اشخاص حقیقی" in s and "تسهیلات" in s:
        return "facilities_person"
    if "اشخاص حقوقی" in s:
        return "facilities_legal"
    if "اشخاص حقیقی" in s:
        # Leaf under حقیقی without product folder — do not invent facilities.
        return None
    return None


def is_cross_product_impact(query: str) -> bool:
    """Cheque + impact/score/facility cues → clarify (never sticky-commit one L2)."""
    q = (query or "").replace("‌", " ")
    if not q:
        return False
    has_cheque = any(k in q for k in ("چک", "صیاد", "برگشت", "پیچک"))
    if not has_cheque:
        return False
    has_impact = any(k in q for k in ("تاثیر", "تأثیر", "اثر"))
    has_scoreish = any(k in q for k in ("امتیاز", "رتبه", "تسهیلات", "اعتبارسنجی", "وام", "قسط"))
    if has_impact and has_scoreish:
        return True
    # Explicit compare both reports
    if ("چکم" in q or "امتیاز چک" in q) and ("تسهیلات" in q or "امتیاز تسهیلات" in q):
        return True
    return False


def parse_subspace_label(ans: str) -> str | None:
    """Parse model output into an L2 id, or None for unclear/ambiguous/unknown."""
    low = (ans or "").strip().lower()
    if not low:
        return None
    if "unclear" in low or "ambiguous" in low:
        return None
    # Prefer longer / more specific ids first
    for label in (
        "facilities_person",
        "facilities_legal",
        "customers_person",
        "customers_legal",
        "supplier_legal",
        "credit_ladder",
        "guarantor",
        "cheque",
        "general",
    ):
        if label in low:
            return label
    if "نردبان" in low:
        return "credit_ladder"
    if "کفیل" in low or "کفالت" in low:
        return "guarantor"
    if "تامین" in low:
        return "supplier_legal"
    if "مشتریان" in low or "پنل" in low:
        if "حقوقی" in low or "شرکت" in low:
            return "customers_legal"
        return "customers_person"
    if "چک" in low or "صیاد" in low or "برگشتی" in low:
        return "cheque"
    if "تسهیلات" in low and ("حقوقی" in low or "شرکت" in low or "سهامدار" in low):
        return "facilities_legal"
    if "تسهیلات" in low and ("حقیقی" in low or "فردی" in low or "افراد" in low):
        return "facilities_person"
    if "عمومی" in low or "معرفی" in low or "هیات مدیره" in low or "کاربرد" in low:
        return "general"
    return None
