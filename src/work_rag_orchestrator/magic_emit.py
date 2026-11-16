"""Deterministic Dana magic-token emission from supplements / clarify state.

Magic is computed POST-HOC after the model answer is final. Widgets are returned
in ``rag.magic`` (structured), NOT embedded into assistant ``content``, so the
plain-text answer is never altered by emission.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

_COLOR_HEX = {
    "آبی": "#26C6DA",
    "ابي": "#26C6DA",
    "سبز": "#43A047",
    "زرد": "#FDD835",
    "نارنجی": "#FB8C00",
    "نارنجي": "#FB8C00",
    "قرمز": "#E53935",
}

_CLARIFY_CARDS: dict[str, dict[str, str]] = {
    "person": {
        "id": "person",
        "label": "گزارش اعتبارسنجی اشخاص حقیقی",
        "subtitle": "اعتبارسنجی وضعیت مالی و اعتباری افراد حقیقی",
        "prompt": "گزارش تسهیلات اشخاص حقیقی",
        "icon": "person",
        "subspace": "facilities_person",
    },
    "legal": {
        "id": "legal",
        "label": "گزارش اعتبارسنجی اشخاص حقوقی",
        "subtitle": "اعتبارسنجی وضعیت مالی اشخاص حقوقی",
        "prompt": "گزارش تسهیلات اشخاص حقوقی / شرکت‌ها",
        "icon": "legal",
        "subspace": "facilities_company",
    },
    "cheque": {
        "id": "cheque",
        "label": "گزارش اعتبارسنجی چک",
        "subtitle": "ارائه گزارش اعتبارسنجی از سوابق چک افراد",
        "prompt": "گزارش چک",
        "icon": "cheque",
        "subspace": "cheque",
    },
}

_MAGIC_RE = re.compile(r"<<<MAGIC:[A-Z][A-Z0-9_]*\s*[\s\S]*?>>>")


def _entity_kind(ent: dict) -> str:
    content = ent.get("content") or {}
    if not isinstance(content, dict):
        return ""
    meta = content.get("_meta") if isinstance(content.get("_meta"), dict) else {}
    return str(meta.get("kind") or "")


def _entity_meta(ent: dict) -> dict[str, Any]:
    content = ent.get("content") or {}
    if not isinstance(content, dict):
        return {}
    meta = content.get("_meta")
    return meta if isinstance(meta, dict) else {}


def _entity_content(ent: dict) -> dict[str, Any]:
    content = ent.get("content") or {}
    return content if isinstance(content, dict) else {}


def _supplemental(state: dict) -> list[dict]:
    raw = state.get("supplemental") or state.get("injected_entities") or []
    return [e for e in raw if isinstance(e, dict)]


def _token(name: str, params: dict[str, Any] | None = None) -> str:
    payload = params or {}
    return f"<<<MAGIC:{name} {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}>>>"


def _color_groups_to_bands(groups: list[Any]) -> list[dict[str, Any]]:
    bands: list[dict[str, Any]] = []
    for g in groups:
        if not isinstance(g, dict):
            continue
        label = str(g.get("رنگ") or g.get("color") or "").strip()
        lo = g.get("حد پایین امتیاز گروه")
        hi = g.get("حد بالای امتیاز گروه")
        if lo is None:
            lo = g.get("score_low")
        if hi is None:
            hi = g.get("score_high")
        try:
            lo_f = float(lo)
            hi_f = float(hi)
        except (TypeError, ValueError):
            continue
        if hi_f < lo_f:
            lo_f, hi_f = hi_f, lo_f
        letter = str(g.get("گروه حرفی") or g.get("letter") or "").strip()
        ranks = g.get("رتبه‌ها") or g.get("ranks") or []
        if not isinstance(ranks, list):
            ranks = []
        codes = [str(x).strip() for x in ranks if str(x).strip()]
        bands.append(
            {
                "label": label or "—",
                "min": int(lo_f) if lo_f == int(lo_f) else lo_f,
                "max": int(hi_f) if hi_f == int(hi_f) else hi_f,
                "color": _COLOR_HEX.get(label, "#90A4AE"),
                "code": letter or (codes[0] if codes else ""),
                "codes": codes,
            }
        )
    # Ladder expects low → high
    bands.sort(key=lambda b: float(b["min"]))
    return bands


def _mask_non_score_numbers(text: str) -> str:
    """Blank years / law-article refs so they cannot be read as ICS scores.

    Real answers often contain «ماده (۵) قانون…» or «سال ۱۳۸۵» — those digits
    must not trip cheque (0–100) / credit (250–900) detectors.
    """
    fa_map = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    t = (text or "").translate(fa_map)
    t = re.sub(r"ماده\s*\(?\s*\d{1,3}\s*\)?", " ", t)
    t = re.sub(r"\b(13|14|19|20)\d{2}\b", " ", t)  # Persian/Gregorian years
    t = re.sub(r"سال\s+\d{2,4}", " ", t)
    return t


def _extract_score(query: str, bands: list[dict[str, Any]], score_min: float, score_max: float) -> float | None:
    """Resolve an explicit score / colour / grade from the user query."""
    if not query:
        return None

    # Latin or Persian digits as a standalone score (years / ماده N masked out)
    q_lat = _mask_non_score_numbers(query)

    # Letter grades A1..E3
    grade_m = re.search(r"\b([A-Ea-e])\s*([1-3])\b", q_lat)
    if grade_m and bands:
        # Prefer fine bands from entity if present via label match later;
        # approximate with color group midpoint when only colour bands exist.
        letter = grade_m.group(1).upper()
        color_by_letter = {"A": "آبی", "B": "سبز", "C": "زرد", "D": "نارنجی", "E": "قرمز"}
        want = color_by_letter.get(letter)
        for b in bands:
            if b.get("label") == want:
                return (float(b["min"]) + float(b["max"])) / 2.0

    for name, _hex in _COLOR_HEX.items():
        if name in query:
            for b in bands:
                if b.get("label") == name or (name in ("ابي",) and b.get("label") == "آبی"):
                    return (float(b["min"]) + float(b["max"])) / 2.0
            # typo ابي → آبی already handled via alias keys matching query

    nums = [float(x) for x in re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)(?![A-Za-z])", q_lat)]
    # Prefer numbers inside the model scale
    in_range = [n for n in nums if score_min <= n <= score_max]
    if in_range:
        # Prefer the most specific (often last) in-range number
        return in_range[-1]
    return None


def _pick_score_entity(ents: list[dict], query: str, subspace: str | None) -> dict | None:
    scores = [e for e in ents if _entity_kind(e) == "score_table"]
    if not scores:
        return None
    q = query or ""
    sub = (subspace or "").lower()

    def rank(e: dict) -> float:
        meta = _entity_meta(e)
        model = str(meta.get("model") or "")
        s = float(e.get("similarity") or 0.0)
        if model == "cheque" and ("چک" in q or "cheque" in sub or "چک" in sub):
            s += 20
        if model == "person" and ("حقیقی" in q or "facilities_person" in sub or "حقیقی" in sub):
            s += 20
        if model == "legal" and ("حقوقی" in q or "facilities_legal" in sub or "حقوقی" in sub or "شرکت" in q):
            s += 20
        # colour-rank default → cheque
        if model == "cheque" and any(c in q for c in ("آبی", "ابي", "سبز", "زرد", "نارنجی", "قرمز", "رتبه")):
            s += 8
        return s

    return max(scores, key=rank)


def _is_horizontal_layout(query: str) -> bool:
    q = query or ""
    return any(k in q for k in ("افقی", "horizontal"))


def _is_vertical_layout(query: str) -> bool:
    q = query or ""
    return "عمودی" in q or "vertical" in (q or "").lower()


def _is_linear_layout(query: str) -> bool:
    q = query or ""
    return any(k in q for k in ("خطی", "linear"))


def _is_ladder_query(query: str) -> bool:
    q = query or ""
    return any(k in q for k in ("نردبان", "ladder"))


def _score_widget(query: str) -> tuple[str, str]:
    """Return (MAGIC token name, variant) for score visualization.

    Concerns are separated by layout keyword — thermometer/vertical bulb is
    only chosen when the user asks for عمودی; linear is its own bar variant.
    """
    if _is_ladder_query(query):
        variant = "horizontal" if _is_horizontal_layout(query) else "vertical"
        return "CREDIT_SCORE_BANDS", variant
    if _is_linear_layout(query):
        return "CREDIT_SCORE_GAUGE", "linear"
    if _is_vertical_layout(query):
        return "CREDIT_SCORE_GAUGE", "vertical"
    if _is_horizontal_layout(query):
        return "CREDIT_SCORE_GAUGE", "horizontal"
    return "CREDIT_SCORE_GAUGE", "ring"


def _infer_model_from_signals(query: str, subspace: str | None, blob: str) -> str:
    """Pick cheque (0–100) vs person/legal (250–900) from explicit signals only.

    Query + subspace win over answer prose — Dana greetings often mention «بررسی
    چک‌ها», which must not force cheque on «امتیاز ۶۱۰ حقیقی».
    """
    q = query or ""
    sub = (subspace or "").lower()
    b = blob or q
    # 1) Explicit query / sticky subspace
    if "چک" in q or "صیاد" in q or "cheque" in sub:
        return "cheque"
    if "حقوقی" in q or "legal" in sub or "facilities_company" in sub or "facilities_legal" in sub:
        return "legal"
    if "حقیقی" in q or "person" in sub or "facilities_person" in sub:
        return "person"
    if "شرکت" in q and ("امتیاز" in q or "رتبه" in q):
        return "legal"
    # 2) Answer-side product wording only when query has no person/legal cue
    if ("چک" in b or "صیاد" in b) and "حقیقی" not in q and "حقوقی" not in q:
        return "cheque"
    # Colour-rank alone defaults to cheque ICS palette
    if any(c in q for c in ("آبی", "ابي", "سبز", "زرد", "نارنجی", "نارنجي", "قرمز")):
        if _blob_has_credit_numeric(q) and not _blob_has_cheque_numeric(q):
            return "person"
        return "cheque"
    if _blob_has_credit_numeric(q) and not _blob_has_cheque_numeric(q):
        return "person"
    if _blob_has_cheque_numeric(q) and not _blob_has_credit_numeric(q):
        return "cheque"
    return "person"


def _score_tokens(ent: dict, query: str, *, answer_blob: str = "") -> list[str]:
    content = _entity_content(ent)
    meta = _entity_meta(ent)
    model = str(meta.get("model") or "person")
    score_min = float(meta.get("score_min") or (1 if model == "cheque" else 250))
    score_max = float(meta.get("score_max") or (100 if model == "cheque" else 900))
    groups = content.get("color_groups") or []
    bands = _color_groups_to_bands(groups if isinstance(groups, list) else [])
    # trim_supplemental may keep only the colour row matching the query
    # (e.g. آبی) — restore the full ICS palette so the gauge still shows
    # the whole ladder while targeting the asked colour/score.
    if model == "cheque":
        defaults = [
            {"label": "قرمز", "min": 1, "max": 40, "color": "#E53935", "code": "E", "codes": ["E1", "E2", "E3"]},
            {"label": "نارنجی", "min": 40, "max": 55, "color": "#FB8C00", "code": "D", "codes": ["D1", "D2", "D3"]},
            {"label": "زرد", "min": 55, "max": 70, "color": "#FDD835", "code": "C", "codes": ["C1", "C2", "C3"]},
            {"label": "سبز", "min": 70, "max": 85, "color": "#43A047", "code": "B", "codes": ["B1", "B2", "B3"]},
            {"label": "آبی", "min": 85, "max": 100, "color": "#26C6DA", "code": "A", "codes": ["A1", "A2", "A3"]},
        ]
        score_min, score_max = 1.0, 100.0
    elif model == "legal":
        defaults = [
            {"label": "قرمز", "min": 250, "max": 459, "color": "#E53935", "code": "E", "codes": ["E1", "E2", "E3"]},
            {"label": "نارنجی", "min": 460, "max": 519, "color": "#FB8C00", "code": "D", "codes": ["D1", "D2", "D3"]},
            {"label": "زرد", "min": 520, "max": 574, "color": "#FDD835", "code": "C", "codes": ["C1", "C2", "C3"]},
            {"label": "سبز", "min": 575, "max": 619, "color": "#43A047", "code": "B", "codes": ["B1", "B2", "B3"]},
            {"label": "آبی", "min": 620, "max": 900, "color": "#26C6DA", "code": "A", "codes": ["A1", "A2", "A3"]},
        ]
        score_min, score_max = 250.0, 900.0
    else:
        defaults = [
            {"label": "قرمز", "min": 250, "max": 459, "color": "#E53935", "code": "E", "codes": ["E1", "E2", "E3"]},
            {"label": "نارنجی", "min": 460, "max": 519, "color": "#FB8C00", "code": "D", "codes": ["D1", "D2", "D3"]},
            {"label": "زرد", "min": 520, "max": 579, "color": "#FDD835", "code": "C", "codes": ["C1", "C2", "C3"]},
            {"label": "سبز", "min": 580, "max": 639, "color": "#43A047", "code": "B", "codes": ["B1", "B2", "B3"]},
            {"label": "آبی", "min": 640, "max": 900, "color": "#26C6DA", "code": "A", "codes": ["A1", "A2", "A3"]},
        ]
        score_min, score_max = 250.0, 900.0
    if len(bands) < 3:
        bands = defaults
    else:
        # Prefer entity mins/maxes when a full palette survived trimming
        score_min = float(meta.get("score_min") or score_min)
        score_max = float(meta.get("score_max") or score_max)
        # Fill missing codes from defaults by colour label
        by_label = {str(d["label"]): d for d in defaults}
        for b in bands:
            if not b.get("code") and not b.get("codes"):
                d = by_label.get(str(b.get("label") or ""))
                if d:
                    b["code"] = d.get("code", "")
                    b["codes"] = list(d.get("codes") or [])

    # Prefer query; fall back to already-generated answer (post-hoc glue).
    score = _extract_score(query, bands, score_min, score_max)
    if score is None and answer_blob:
        score = _extract_score(answer_blob, bands, score_min, score_max)

    # Active band label / codes for gauge
    band_label = ""
    band_code = ""
    band_codes: list[str] = []
    for b in bands:
        if score is not None and float(b["min"]) <= score <= float(b["max"]):
            band_label = str(b["label"])
            band_code = str(b.get("code") or "")
            raw_codes = b.get("codes") or []
            band_codes = [str(x) for x in raw_codes if str(x).strip()]
            break

    # If no explicit score, park needle at mid of active colour band or scale
    if score is None:
        if band_label:
            for b in bands:
                if str(b["label"]) == band_label:
                    score = (float(b["min"]) + float(b["max"])) / 2.0
                    break
        if score is None:
            score = (score_min + score_max) / 2.0
            for b in bands:
                if float(b["min"]) <= score <= float(b["max"]):
                    band_label = str(b["label"])
                    band_code = str(b.get("code") or "")
                    band_codes = [str(x) for x in (b.get("codes") or []) if str(x).strip()]
                    break

    label = {
        "cheque": "امتیاز اعتبارسنجی چک",
        "person": "امتیاز اعتباری حقیقی",
        "legal": "امتیاز اعتباری حقوقی",
    }.get(model, "امتیاز اعتباری")

    params_common = {
        "score": round(score, 1) if isinstance(score, float) and score != int(score) else int(score),
        "min": int(score_min),
        "max": int(score_max),
        "label": label,
        "band": band_label,
        "code": band_code,
        "codes": band_codes,
        "model": model,
        "bands": bands,
        "color_mode": "stepped",
    }

    token_name, variant = _score_widget(query)
    return [_token(token_name, {**params_common, "variant": variant})]


def _clarify_items(state: dict) -> list[dict[str, str]]:
    """ICS clarify cards must mirror the clarifying question — not dump every report type.

    Dumping cheque+person+legal into every clarify polluted downstream routers
    (stress harness / sticky pickers) because the MAGIC JSON always contained «چک».
    """
    query = state.get("query") or ""
    # Prefer the clean clarifying question (set before magic append).
    clarify_q = state.get("clarifying_question") or ""
    if "<<<MAGIC:" in clarify_q:
        clarify_q = _MAGIC_RE.sub("", clarify_q)
    available = state.get("available_subspaces") or []

    # Colour-rank clarify → چک vs تسهیلات (not all three report products)
    try:
        from .subspace_routing import query_has_cheque_color_rank

        if query_has_cheque_color_rank(query) or (
            "چک" in clarify_q and "تسهیلات" in clarify_q
        ):
            return [_CLARIFY_CARDS["cheque"], _CLARIFY_CARDS["person"]]
    except Exception:
        if "چک" in clarify_q and "تسهیلات" in clarify_q:
            return [_CLARIFY_CARDS["cheque"], _CLARIFY_CARDS["person"]]

    # حقیقی vs حقوقی fork — do NOT include cheque card
    if "حقیقی" in clarify_q and "حقوقی" in clarify_q:
        return [_CLARIFY_CARDS["person"], _CLARIFY_CARDS["legal"]]

    # شخصی / شرکتی wording
    if "شخصی" in clarify_q and ("شرکتی" in clarify_q or "حقوقی" in clarify_q):
        return [_CLARIFY_CARDS["person"], _CLARIFY_CARDS["legal"]]

    keys: list[str] = []
    # Use query + clarify text only (NOT the full available_subspaces dump —
    # that list often contains every product label and forced all three cards).
    blob = f"{query} {clarify_q}"
    if any(k in blob for k in ("چک", "cheque", "صیاد")):
        keys.append("cheque")
    if any(k in blob for k in ("حقیقی", "افراد", "شخصی", "facilities_person")):
        keys.append("person")
    if any(k in blob for k in ("حقوقی", "شرکت", "شرکتی", "facilities_legal")):
        keys.append("legal")

    # Generic report-type clarify with no fork signal → all three
    if not keys and any(
        k in (clarify_q + query) for k in ("گزارش", "نوع", "کدام", "مایلید", "بفرمایید")
    ):
        keys = ["person", "legal", "cheque"]

    # Fall back to available labels only when still empty
    if not keys:
        avail_blob = " ".join(str(x) for x in available)
        if any(k in avail_blob for k in ("چک", "cheque")):
            keys.append("cheque")
        if any(k in avail_blob for k in ("حقیقی", "افراد", "facilities_person")):
            keys.append("person")
        if any(k in avail_blob for k in ("حقوقی", "شرکت", "facilities_legal")):
            keys.append("legal")

    if not keys:
        keys = ["person", "legal", "cheque"]

    order = ["person", "legal", "cheque"]
    out = [_CLARIFY_CARDS[k] for k in order if k in keys]
    return out or [_CLARIFY_CARDS["person"], _CLARIFY_CARDS["legal"]]


def _history_query(query: str, answer: str = "") -> bool:
    """ICS company history / intro → COMPANY_TIMELINE.

    Matches explicit تاریخچه/تأسیس asks, ICS «چیست/معرفی» intros, and
    founding language already present in the generated answer.
    """
    q = query or ""
    a = answer or ""
    if any(
        k in q
        for k in (
            "تاریخچه",
            "تاریخ شرکت",
            "تأسیس",
            "تاسیس",
            "سابقه شرکت اعتبارسنجی",
            "خط زمان",
            "تایم‌لاین",
            "تایم لاین",
            "timeline",
        )
    ) or ("تاریخ" in q and "شرکت" in q and "اعتبارسنجی" in q):
        return True
    if _ics_company_query(q):
        if any(
            k in q
            for k in (
                "چیست",
                "چی هست",
                "کیست",
                "معرفی",
                "درباره",
                "توضیح",
                "که هست",
                "چه شرکتی",
            )
        ):
            return True
        if any(
            k in a
            for k in (
                "تأسیس",
                "تاسیس",
                "تاریخچه",
                "آبان",
                "سال ۱۳۸",
                "سال 138",
                "ماده",
            )
        ):
            return True
    return False


def _ics_company_query(query: str) -> bool:
    """True when the ask is about شرکت اعتبارسنجی ایران specifically."""
    q = query or ""
    return ("اعتبارسنجی ایران" in q) or ("شرکت اعتبارسنجی" in q) or ("ICS" in q.upper())


def _shareholders_query(query: str) -> bool:
    q = query or ""
    if not any(k in q for k in ("سهامدار", "سهام‌دار", "سهام دار")):
        return False
    # Do not fire on generic credit-scoring asks — require ICS company context.
    return _ics_company_query(q)


def _creditors_query(query: str) -> bool:
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
            "تامین‌کنندگان دیتا",
            "تأمین‌کنندگان داده",
            "تامین کننده اطلاعات",
            "تأمین کننده اطلاعات",
        )
    )
    if not hit:
        if ("داده" in q or "دیتا" in q) and any(
            k in q for k in ("بانک", "نئوبانک", "لیزینگ", "لندتک", "بیمه", "صندوق")
        ):
            hit = True
    if not hit:
        return False
    return _ics_company_query(q)


def _best_kind_sim(ents: list[dict], kind: str) -> float:
    sims = [float(e.get("similarity") or 0.0) for e in ents if _entity_kind(e) == kind]
    return max(sims) if sims else -1.0


def build_magic_tokens(state: dict, *, finish_reason: str | None = None) -> list[str]:
    """Return magic token strings to append (may be empty)."""
    if state.get("blocked") or state.get("greeting_only"):
        return []
    if finish_reason == "content_filter":
        return []

    query = state.get("query") or state.get("rewritten_query") or ""
    ents = _supplemental(state)
    tokens: list[str] = []

    # Clarify companion / primary
    if finish_reason == "clarify" or state.get("needs_clarification"):
        items = _clarify_items(state)
        tokens.append(
            _token(
                "CLARIFY_OPTIONS",
                {
                    "title": "لطفاً نوع گزارش را انتخاب کنید",
                    "items": items,
                },
            )
        )
        return tokens  # clarify turn: only chips

    subspace = state.get("applied_subspace") or state.get("sticky_subspace")

    # Score tables → gauge / bands. Emit only when the QUERY has score intent
    # (never on generic «شرکت اعتبارسنجی ایران چیست؟» / answer-side ماده ۵).
    answer_blob = str(state.get("_magic_answer_blob") or "")
    signal_blob = f"{query}\n{answer_blob}"
    if "_magic_specific_score" in state:
        specific = bool(state.get("_magic_specific_score"))
    else:
        specific = _query_has_score_intent(query)

    score_ent = _pick_score_entity(ents, query, subspace if isinstance(subspace, str) else None)
    # If Entity Data missed the score table but the turn has a specific score
    # signal, synthesize ICS defaults for the inferred model family.
    if not score_ent and specific:
        model = _infer_model_from_signals(
            query, subspace if isinstance(subspace, str) else None, signal_blob
        )
        score_ent = {
            "similarity": 0.0,
            "list_name": f"synthetic-{model}",
            "content": {
                "_meta": {
                    "kind": "score_table",
                    "model": model,
                    "score_min": 1 if model == "cheque" else 250,
                    "score_max": 100 if model == "cheque" else 900,
                },
                "color_groups": [],
            },
        }
    if score_ent and specific:
        # Override entity model when query/answer clearly points at another scale
        inferred = _infer_model_from_signals(
            query, subspace if isinstance(subspace, str) else None, signal_blob
        )
        ent_model = str(_entity_meta(score_ent).get("model") or "")
        if inferred and ent_model and inferred != ent_model:
            # Prefer explicit cheque/credit signals over co-retrieved table
            if (inferred == "cheque" and _blob_has_cheque_numeric(signal_blob)) or (
                inferred != "cheque" and _blob_has_credit_numeric(signal_blob)
            ):
                meta = dict(_entity_meta(score_ent))
                meta["model"] = inferred
                meta["score_min"] = 1 if inferred == "cheque" else 250
                meta["score_max"] = 100 if inferred == "cheque" else 900
                content = dict(_entity_content(score_ent))
                content["_meta"] = meta
                score_ent = {**score_ent, "content": content}
        tokens.extend(_score_tokens(score_ent, query, answer_blob=answer_blob))

    # Shareholders / creditors — explicit query intent only (never co-retrieval).
    want_sh = _shareholders_query(query)
    want_cr = _creditors_query(query)
    if want_sh:
        tokens.append(_token("SHAREHOLDERS", {"title": "سهامداران شرکت اعتبارسنجی ایران"}))
    if want_cr:
        tokens.append(
            _token(
                "CREDITORS",
                {"title": "تأمین‌کنندگان داده‌های شرکت اعتبارسنجی ایران"},
            )
        )

    # Company history timeline (chunk-backed; no entity kind required)
    if _history_query(query, answer_blob):
        tokens.append(_token("COMPANY_TIMELINE", {"variant": "vertical"}))
        # Company intro/history turns must not also glue a co-retrieved score
        # gauge (e.g. «ماده (۵)» misread as cheque score 5).
        if not _query_has_score_intent(query):
            tokens = [t for t in tokens if "CREDIT_SCORE_GAUGE" not in t and "CREDIT_SCORE_BANDS" not in t]

    return tokens


def strip_magic_from_content(content: str) -> str:
    """Remove any MAGIC blocks from assistant text (defense in depth)."""
    text = _MAGIC_RE.sub("", content or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_token_string(token: str) -> dict[str, Any] | None:
    m = re.match(r"<<<MAGIC:([A-Z][A-Z0-9_]*)\s*(\{.*\})?\s*>>>", token or "", re.S)
    if not m:
        return None
    name = m.group(1)
    params: dict[str, Any] = {}
    raw = (m.group(2) or "").strip()
    if raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                params = obj
        except json.JSONDecodeError:
            params = {}
    return {"name": name, "params": params}


def _blob_has_cheque_numeric(blob: str) -> bool:
    q = _mask_non_score_numbers(blob)
    for m in re.finditer(r"(?<![.\d])(\d{1,3})(?![.\d])", q):
        n = int(m.group(1))
        if 0 <= n <= 100:
            return True
    return False


def _blob_has_credit_numeric(blob: str) -> bool:
    q = _mask_non_score_numbers(blob)
    for m in re.finditer(r"(?<![.\d])(\d{3})(?![.\d])", q):
        n = int(m.group(1))
        if 250 <= n <= 900:
            return True
    return False


def _blob_has_colour_tag(blob: str) -> bool:
    return any(c in (blob or "") for c in ("آبی", "ابي", "سبز", "زرد", "نارنجی", "نارنجي", "قرمز"))


def _blob_has_named_band(blob: str) -> bool:
    return any(
        k in (blob or "")
        for k in ("عالی", "خوب", "متوسط", "ضعیف", "قابل قبول", "ریسک خیلی پایین", "ریسک پایین")
    )


def _blob_has_grade_code(blob: str) -> bool:
    return bool(re.search(r"\b[A-Ea-e]\s*[1-3]\b", blob or ""))


def _query_has_score_intent(query: str) -> bool:
    """Score widgets require an explicit ask in the user query — never answer prose alone.

    Incidental digits in answers (ماده ۵، سال ۱۳۸۵، «امتیاز چک» product lists) must
    not invent a CREDIT_SCORE_GAUGE on company-overview turns.
    """
    q = query or ""
    if not q.strip():
        return False
    if any(
        k in q
        for k in (
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
    ):
        return True
    if _blob_has_colour_tag(q) or _blob_has_named_band(q) or _blob_has_grade_code(q):
        return True
    if _blob_has_cheque_numeric(q) or _blob_has_credit_numeric(q):
        return True
    return False


def collect_magic_payload(
    state: dict,
    content: str = "",
    *,
    finish_reason: str | None = None,
) -> list[dict[str, Any]]:
    """Post-hoc magic items for ``rag.magic`` — never mutates content.

    Score gauges gate on the **query** (explicit score/colour/name/layout).
    ICS timeline/shareholders gate on company intent (+ answer founding cues).
    """
    if state.get("blocked") or state.get("greeting_only"):
        return []
    fr = finish_reason or (state.get("formatted_response") or {}).get("finish_reason")
    if fr == "content_filter":
        return []

    query = state.get("query") or state.get("rewritten_query") or ""
    answer = strip_magic_from_content(content or state.get("answer") or "")
    blob = f"{query}\n{answer}"

    # Query-only score gate — answer digits/colours never open a gauge alone.
    specific_score = _query_has_score_intent(query)

    prev_answer = state.get("_magic_answer_blob")
    state["_magic_answer_blob"] = blob
    state["_magic_specific_score"] = specific_score
    try:
        built = build_magic_tokens(state, finish_reason=fr)
    except Exception as e:
        log.warning("magic_emit collect failed: %s", e)
        built = []
    finally:
        if prev_answer is None:
            state.pop("_magic_answer_blob", None)
        else:
            state["_magic_answer_blob"] = prev_answer
        state.pop("_magic_specific_score", None)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for t in built:
        item = _parse_token_string(t)
        if not item:
            continue
        name = item["name"]
        if name in seen:
            continue
        # Drop score widgets when the turn has no specific score signal
        if name in ("CREDIT_SCORE_GAUGE", "CREDIT_SCORE_BANDS") and not specific_score:
            continue
        seen.add(name)
        out.append(item)
    state["magic_payload"] = out
    return out


def append_magic_tokens(content: str, state: dict, *, finish_reason: str | None = None) -> str:
    """Backward-compatible entry: strip MAGIC from body and stash payload on state.

    Does NOT append MAGIC into content anymore — callers must send
    ``state['magic_payload']`` via ``rag.magic``.
    """
    text = strip_magic_from_content(content or "")
    if not text:
        collect_magic_payload(state, "", finish_reason=finish_reason)
        return content or ""

    query = state.get("query") or state.get("rewritten_query") or ""
    score_viz = any(
        k in query
        for k in ("امتیاز", "رتبه", "رنگ", "نردبان", "عمودی", "افقی", "آبی", "سبز", "زرد", "نارنجی", "قرمز")
    )
    if "یافت نشد" in text and len(text) < 160 and not score_viz:
        state["magic_payload"] = []
        return text

    fr = finish_reason or (state.get("formatted_response") or {}).get("finish_reason")
    collect_magic_payload(state, text, finish_reason=fr)
    return text
