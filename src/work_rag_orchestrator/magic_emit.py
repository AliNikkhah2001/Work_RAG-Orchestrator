"""Deterministic Dana magic-token emission from supplements / clarify state.

The LLM answers in plain Persian. Open WebUI parses ``<<<MAGIC:TOKEN {json}>>>``
blocks from assistant content. This module appends those tokens after the
text answer so widgets (gauges, timeline, shareholders, creditors, clarify)
render automatically.
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
        bands.append(
            {
                "label": label or "—",
                "min": int(lo_f) if lo_f == int(lo_f) else lo_f,
                "max": int(hi_f) if hi_f == int(hi_f) else hi_f,
                "color": _COLOR_HEX.get(label, "#90A4AE"),
            }
        )
    # Ladder expects low → high
    bands.sort(key=lambda b: float(b["min"]))
    return bands


def _extract_score(query: str, bands: list[dict[str, Any]], score_min: float, score_max: float) -> float | None:
    """Resolve an explicit score / colour / grade from the user query."""
    if not query:
        return None

    # Latin or Persian digits as a standalone score
    fa_map = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    q_lat = query.translate(fa_map)

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
    return "عمودی" in q


def _is_ladder_query(query: str) -> bool:
    q = query or ""
    return any(k in q for k in ("نردبان", "ladder"))


def _score_widget(query: str) -> tuple[str, str]:
    """Return (MAGIC token name, variant) for score visualization."""
    if _is_ladder_query(query):
        variant = "horizontal" if _is_horizontal_layout(query) else "vertical"
        return "CREDIT_SCORE_BANDS", variant
    if _is_vertical_layout(query):
        return "CREDIT_SCORE_GAUGE", "vertical"
    if _is_horizontal_layout(query):
        return "CREDIT_SCORE_GAUGE", "horizontal"
    return "CREDIT_SCORE_GAUGE", "ring"


def _score_tokens(ent: dict, query: str) -> list[str]:
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
            {"label": "قرمز", "min": 1, "max": 40, "color": "#E53935"},
            {"label": "نارنجی", "min": 40, "max": 55, "color": "#FB8C00"},
            {"label": "زرد", "min": 55, "max": 70, "color": "#FDD835"},
            {"label": "سبز", "min": 70, "max": 85, "color": "#43A047"},
            {"label": "آبی", "min": 85, "max": 100, "color": "#26C6DA"},
        ]
        score_min, score_max = 1.0, 100.0
    elif model == "legal":
        defaults = [
            {"label": "قرمز", "min": 250, "max": 459, "color": "#E53935"},
            {"label": "نارنجی", "min": 460, "max": 519, "color": "#FB8C00"},
            {"label": "زرد", "min": 520, "max": 574, "color": "#FDD835"},
            {"label": "سبز", "min": 575, "max": 619, "color": "#43A047"},
            {"label": "آبی", "min": 620, "max": 900, "color": "#26C6DA"},
        ]
        score_min, score_max = 250.0, 900.0
    else:
        defaults = [
            {"label": "قرمز", "min": 250, "max": 459, "color": "#E53935"},
            {"label": "نارنجی", "min": 460, "max": 519, "color": "#FB8C00"},
            {"label": "زرد", "min": 520, "max": 579, "color": "#FDD835"},
            {"label": "سبز", "min": 580, "max": 639, "color": "#43A047"},
            {"label": "آبی", "min": 640, "max": 900, "color": "#26C6DA"},
        ]
        score_min, score_max = 250.0, 900.0
    if len(bands) < 3:
        bands = defaults
    else:
        # Prefer entity mins/maxes when a full palette survived trimming
        score_min = float(meta.get("score_min") or score_min)
        score_max = float(meta.get("score_max") or score_max)

    score = _extract_score(query, bands, score_min, score_max)
    # If no explicit score, park needle at mid of scale (still shows bands)
    if score is None:
        score = (score_min + score_max) / 2.0

    # Active band label for gauge
    band_label = ""
    for b in bands:
        if float(b["min"]) <= score <= float(b["max"]):
            band_label = str(b["label"])
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
        "model": model,
        "bands": bands,
    }

    token_name, variant = _score_widget(query)
    return [_token(token_name, {**params_common, "variant": variant})]


def _clarify_items(state: dict) -> list[dict[str, str]]:
    query = state.get("query") or ""
    available = state.get("available_subspaces") or []

    # Colour-rank clarify → چک vs تسهیلات (حقیقی card covers facilities path)
    try:
        from .subspace_routing import query_has_cheque_color_rank

        if query_has_cheque_color_rank(query):
            return [_CLARIFY_CARDS["cheque"], _CLARIFY_CARDS["person"], _CLARIFY_CARDS["legal"]]
    except Exception:
        pass

    keys: list[str] = []
    blob = " ".join(str(x) for x in available) + " " + query
    if any(k in blob for k in ("چک", "cheque", "گزارش چک")):
        keys.append("cheque")
    if any(k in blob for k in ("حقیقی", "افراد", "facilities_person", "تسهیلات افراد")):
        keys.append("person")
    if any(k in blob for k in ("حقوقی", "شرکت", "facilities_legal", "تسهیلات شرکت")):
        keys.append("legal")

    # Default three report types when clarify is generic
    if not keys:
        keys = ["person", "legal", "cheque"]

    # Preserve order person → legal → cheque
    order = ["person", "legal", "cheque"]
    out = [_CLARIFY_CARDS[k] for k in order if k in keys]
    return out or list(_CLARIFY_CARDS.values())


def _history_query(query: str) -> bool:
    q = query or ""
    return any(
        k in q
        for k in (
            "تاریخچه",
            "تاریخ شرکت",
            "تأسیس شرکت",
            "تاسیس شرکت",
            " chronologie",  # never
            "سابقه شرکت اعتبارسنجی",
            "خط زمان",
        )
    ) or ("تاریخ" in q and "شرکت" in q and "اعتبارسنجی" in q)


def _shareholders_query(query: str) -> bool:
    return any(k in (query or "") for k in ("سهامدار", "سهام‌دار", "سهام دار"))


def _creditors_query(query: str) -> bool:
    q = query or ""
    if any(
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
    ):
        return True
    # «چه بانک‌هایی داده می‌دهند» style
    if ("داده" in q or "دیتا" in q) and any(
        k in q for k in ("بانک", "نئوبانک", "لیزینگ", "لندتک", "بیمه", "صندوق")
    ):
        return True
    return False


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

    kinds = {_entity_kind(e) for e in ents}
    subspace = state.get("applied_subspace") or state.get("sticky_subspace")

    # Score tables → gauge / bands
    score_ent = _pick_score_entity(ents, query, subspace if isinstance(subspace, str) else None)
    score_intent = any(
        k in query
        for k in (
            "امتیاز",
            "رتبه",
            "رنگ",
            "گرید",
            "نردبان",
            "A1",
            "B1",
            "آبی",
            "ابي",
            "سبز",
            "زرد",
            "نارنجی",
            "قرمز",
            "جدول امتیاز",
            "عمودی",
            "افقی",
        )
    )
    # If Entity Data missed the score table but the user clearly asks for a
    # score/colour/layout gauge, synthesize ICS defaults for the inferred model.
    if not score_ent and score_intent:
        model = "person"
        if "چک" in query or (isinstance(subspace, str) and "cheque" in subspace):
            model = "cheque"
        elif "حقوقی" in query or "شرکت" in query or (
            isinstance(subspace, str) and "legal" in subspace
        ):
            model = "legal"
        elif "حقیقی" in query or (
            isinstance(subspace, str) and "person" in subspace
        ):
            model = "person"
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
    if score_ent and (score_intent or "score_table" in kinds):
        # Avoid attaching score gauge on pure catalog questions
        catalog_only = any(k in query for k in ("سهامدار", "تامین", "تأمین", "وام", "عناوین")) and not score_intent
        if not catalog_only:
            tokens.extend(_score_tokens(score_ent, query))

    # Shareholders / creditors — query intent wins; otherwise pick the
    # higher-similarity catalog so co-retrieved lists don't double-emit.
    want_sh = _shareholders_query(query)
    want_cr = _creditors_query(query)
    sh_sim = _best_kind_sim(ents, "shareholders")
    cr_sim = _best_kind_sim(ents, "data_providers")
    emit_sh = False
    emit_cr = False
    if want_sh and not want_cr:
        emit_sh = True
    elif want_cr and not want_sh:
        emit_cr = True
    elif want_sh and want_cr:
        emit_sh = emit_cr = True
    elif not score_intent:
        if sh_sim >= 0 and sh_sim >= cr_sim:
            emit_sh = True
        if cr_sim >= 0 and cr_sim > sh_sim:
            emit_cr = True
    if emit_sh:
        tokens.append(_token("SHAREHOLDERS", {"title": "سهامداران شرکت اعتبارسنجی ایران"}))
    if emit_cr:
        tokens.append(
            _token(
                "CREDITORS",
                {"title": "تأمین‌کنندگان داده‌های شرکت اعتبارسنجی ایران"},
            )
        )

    # Company history timeline (chunk-backed; no entity kind required)
    if _history_query(query):
        tokens.append(_token("COMPANY_TIMELINE", {"variant": "vertical"}))

    return tokens


def append_magic_tokens(content: str, state: dict, *, finish_reason: str | None = None) -> str:
    """Append magic tokens to assistant content (idempotent)."""
    text = (content or "").rstrip()
    if not text:
        return content or ""

    query = state.get("query") or state.get("rewritten_query") or ""
    score_viz = any(
        k in query
        for k in ("امتیاز", "رتبه", "رنگ", "نردبان", "عمودی", "افقی", "آبی", "سبز", "زرد", "نارنجی", "قرمز")
    )
    # Skip plain refusals unless the user asked for a score visualization —
    # then still attach the gauge so the UI can render the asked number/band.
    if "یافت نشد" in text and len(text) < 160 and not score_viz:
        return text

    existing = set(_MAGIC_RE.findall(text))
    try:
        fr = finish_reason or (state.get("formatted_response") or {}).get("finish_reason")
        built = build_magic_tokens(state, finish_reason=fr)
    except Exception as e:
        log.warning("magic_emit failed: %s", e)
        return text

    present_names = {m.group(1) for m in re.finditer(r"<<<MAGIC:([A-Z][A-Z0-9_]*)", text)}
    extras: list[str] = []
    for t in built:
        m = re.match(r"<<<MAGIC:([A-Z][A-Z0-9_]*)", t)
        if not m:
            continue
        if m.group(1) in present_names or t in existing:
            continue
        extras.append(t)
        present_names.add(m.group(1))

    if not extras:
        return text
    return text + "\n\n" + "\n".join(extras)
