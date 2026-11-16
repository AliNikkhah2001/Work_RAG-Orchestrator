"""Unit tests for deterministic Dana magic-token emission."""

from __future__ import annotations

import json
import re

from work_rag_orchestrator.magic_emit import (
    append_magic_tokens,
    build_magic_tokens,
    _extract_score,
    _color_groups_to_bands,
)


def _score_ent(model: str, groups: list[dict], sim: float = 0.9) -> dict:
    return {
        "similarity": sim,
        "list_name": f"score-{model}",
        "content": {
            "_meta": {
                "kind": "score_table",
                "model": model,
                "score_min": 1 if model == "cheque" else 250,
                "score_max": 100 if model == "cheque" else 900,
            },
            "color_groups": groups,
        },
    }


CHEQUE_GROUPS = [
    {"رنگ": "آبی", "حد پایین امتیاز گروه": 85.0, "حد بالای امتیاز گروه": 100.0},
    {"رنگ": "سبز", "حد پایین امتیاز گروه": 70.0, "حد بالای امتیاز گروه": 85.0},
    {"رنگ": "زرد", "حد پایین امتیاز گروه": 55.0, "حد بالای امتیاز گروه": 70.0},
    {"رنگ": "نارنجی", "حد پایین امتیاز گروه": 40.0, "حد بالای امتیاز گروه": 55.0},
    {"رنگ": "قرمز", "حد پایین امتیاز گروه": 1.0, "حد بالای امتیاز گروه": 40.0},
]

PERSON_GROUPS = [
    {"رنگ": "آبی", "حد پایین امتیاز گروه": 640.0, "حد بالای امتیاز گروه": 900.0},
    {"رنگ": "سبز", "حد پایین امتیاز گروه": 580.0, "حد بالای امتیاز گروه": 639.0},
    {"رنگ": "زرد", "حد پایین امتیاز گروه": 520.0, "حد بالای امتیاز گروه": 579.0},
    {"رنگ": "نارنجی", "حد پایین امتیاز گروه": 460.0, "حد بالای امتیاز گروه": 519.0},
    {"رنگ": "قرمز", "حد پایین امتیاز گروه": 250.0, "حد بالای امتیاز گروه": 459.0},
]


def _parse_params(token: str) -> dict:
    start = token.find("{")
    end = token.rfind("}")
    assert start >= 0 and end > start
    return json.loads(token[start : end + 1])


def test_cheque_gauge_uses_1_100_and_color_score():
    state = {
        "query": "رتبه آبی چک یعنی چه امتیاز؟",
        "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)],
        "applied_subspace": "cheque",
    }
    toks = build_magic_tokens(state, finish_reason="stop")
    assert len(toks) == 1
    assert toks[0].startswith("<<<MAGIC:CREDIT_SCORE_GAUGE")
    p = _parse_params(toks[0])
    assert p["min"] == 1
    assert p["max"] == 100
    assert p["model"] == "cheque"
    assert p["band"] == "آبی"
    assert 85 <= float(p["score"]) <= 100


def test_person_gauge_targets_explicit_score():
    state = {
        "query": "امتیاز ۶۱۰ برای حقیقی در چه رنگی است؟",
        "supplemental": [_score_ent("person", PERSON_GROUPS)],
        "applied_subspace": "facilities_person",
    }
    toks = build_magic_tokens(state, finish_reason="stop")
    p = _parse_params(toks[0])
    assert p["min"] == 250
    assert p["max"] == 900
    assert p["score"] == 610
    assert p["band"] == "سبز"


def test_ladder_emits_bands_token():
    state = {
        "query": "نردبان اعتباری حقیقی چیست؟",
        "supplemental": [_score_ent("person", PERSON_GROUPS)],
    }
    toks = build_magic_tokens(state, finish_reason="stop")
    assert toks[0].startswith("<<<MAGIC:CREDIT_SCORE_BANDS")
    p = _parse_params(toks[0])
    assert p["variant"] == "vertical"
    assert p["model"] == "person"


def test_ladder_horizontal_when_query_requests():
    state = {
        "query": "نردبان افقی اعتباری حقوقی",
        "supplemental": [_score_ent("legal", PERSON_GROUPS)],
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert p["variant"] == "horizontal"
    assert p["model"] == "legal"
    assert p["min"] == 250 and p["max"] == 900


def test_ladder_horizontal_english_keyword():
    state = {
        "query": "credit ladder horizontal for cheque",
        "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)],
        "applied_subspace": "cheque",
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert p["variant"] == "horizontal"
    assert p["model"] == "cheque"


def test_gauge_vertical_variant():
    state = {
        "query": "امتیاز اعتباری به صورت عمودی نشان بده",
        "supplemental": [_score_ent("person", PERSON_GROUPS)],
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert build_magic_tokens(state, finish_reason="stop")[0].startswith("<<<MAGIC:CREDIT_SCORE_GAUGE")
    assert p["variant"] == "vertical"
    assert p["model"] == "person"


def test_gauge_horizontal_variant():
    state = {
        "query": "گرید افقی امتیاز چک",
        "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)],
        "applied_subspace": "cheque",
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert build_magic_tokens(state, finish_reason="stop")[0].startswith("<<<MAGIC:CREDIT_SCORE_GAUGE")
    assert p["variant"] == "horizontal"
    assert p["model"] == "cheque"


def test_default_gauge_ring_variant():
    state = {
        "query": "رتبه سبز چک چقدر است؟",
        "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)],
        "applied_subspace": "cheque",
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert build_magic_tokens(state, finish_reason="stop")[0].startswith("<<<MAGIC:CREDIT_SCORE_GAUGE")
    assert p["variant"] == "ring"
    assert p["model"] == "cheque"


def test_person_gauge_includes_model_field():
    state = {
        "query": "امتیاز ۵۵۰ حقیقی",
        "supplemental": [_score_ent("person", PERSON_GROUPS)],
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert p["model"] == "person"
    assert p["variant"] == "ring"


def test_shareholders_and_creditors():
    sh = {
        "similarity": 0.9,
        "content": {"_meta": {"kind": "shareholders"}, "records": []},
    }
    pr = {
        "similarity": 0.9,
        "content": {"_meta": {"kind": "data_providers"}, "records": []},
    }
    assert any(
        t.startswith("<<<MAGIC:SHAREHOLDERS")
        for t in build_magic_tokens(
            {"query": "سهامداران شرکت چه کسانی هستند؟", "supplemental": [sh]},
            finish_reason="stop",
        )
    )
    assert any(
        t.startswith("<<<MAGIC:CREDITORS")
        for t in build_magic_tokens(
            {"query": "تامین‌کنندگان دیتا کدام‌اند؟", "supplemental": [pr]},
            finish_reason="stop",
        )
    )


def test_timeline_on_history_query():
    toks = build_magic_tokens(
        {"query": "تاریخچه شرکت اعتبارسنجی ایران را بگو", "supplemental": []},
        finish_reason="stop",
    )
    assert any(t.startswith("<<<MAGIC:COMPANY_TIMELINE") for t in toks)


def test_clarify_emits_cards():
    toks = build_magic_tokens(
        {
            "query": "وضعیت اعتباری چطور است",
            "needs_clarification": True,
            "available_subspaces": ["گزارش چک", "تسهیلات افراد", "تسهیلات شرکت‌ها"],
        },
        finish_reason="clarify",
    )
    assert len(toks) == 1
    assert toks[0].startswith("<<<MAGIC:CLARIFY_OPTIONS")
    p = _parse_params(toks[0])
    ids = [i["id"] for i in p["items"]]
    assert ids == ["person", "legal", "cheque"]


def test_append_idempotent():
    state = {
        "query": "تاریخچه شرکت اعتبارسنجی ایران",
        "supplemental": [],
    }
    once = append_magic_tokens("متن پاسخ", state, finish_reason="stop")
    twice = append_magic_tokens(once, state, finish_reason="stop")
    assert once.count("COMPANY_TIMELINE") == 1
    assert twice.count("COMPANY_TIMELINE") == 1


def test_extract_score_persian_digits():
    bands = _color_groups_to_bands(PERSON_GROUPS)
    assert _extract_score("امتیاز ۷۲۰", bands, 250, 900) == 720.0


def test_truncated_color_groups_restored():
    """trim_supplemental may keep only the matching colour row."""
    state = {
        "query": "رتبه آبی چک",
        "supplemental": [
            _score_ent(
                "cheque",
                [{"رنگ": "آبی", "حد پایین امتیاز گروه": 85.0, "حد بالای امتیاز گروه": 100.0}],
            )
        ],
        "applied_subspace": "cheque",
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert len(p["bands"]) == 5
    assert p["band"] == "آبی"
    assert p["min"] == 1 and p["max"] == 100
    assert p["model"] == "cheque"


def test_truncated_legal_color_groups_restored():
    state = {
        "query": "رتبه قرمز حقوقی",
        "supplemental": [
            _score_ent(
                "legal",
                [{"رنگ": "قرمز", "حد پایین امتیاز گروه": 250.0, "حد بالای امتیاز گروه": 459.0}],
            )
        ],
        "applied_subspace": "facilities_company",
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert len(p["bands"]) == 5
    assert p["model"] == "legal"
    assert p["min"] == 250 and p["max"] == 900


def test_shareholders_does_not_emit_creditors():
    state = {
        "query": "سهامداران شرکت اعتبارسنجی ایران چه کسانی هستند؟",
        "supplemental": [
            {"similarity": 0.9, "content": {"_meta": {"kind": "shareholders"}}},
            {"similarity": 0.7, "content": {"_meta": {"kind": "data_providers"}}},
        ],
    }
    names = [re.match(r"<<<MAGIC:([A-Z_]+)", t).group(1) for t in build_magic_tokens(state, finish_reason="stop")]
    assert names == ["SHAREHOLDERS"]


def test_greeting_skips_magic():
    toks = build_magic_tokens(
        {"query": "سلام روز بخیر", "greeting_only": "greeting", "needs_clarification": False},
        finish_reason="stop",
    )
    assert toks == []


def test_score_table_without_score_intent_emits_nothing():
    """Co-retrieved score tables must not force a gauge."""
    toks = build_magic_tokens(
        {
            "query": "گزارش تسهیلات چیست",
            "supplemental": [_score_ent("person", PERSON_GROUPS)],
        },
        finish_reason="stop",
    )
    assert toks == []


def test_shareholders_not_emitted_from_similarity_alone():
    toks = build_magic_tokens(
        {
            "query": "گزارش اعتباری چیست",
            "supplemental": [
                {"similarity": 0.99, "content": {"_meta": {"kind": "shareholders"}}},
            ],
        },
        finish_reason="stop",
    )
    assert toks == []
