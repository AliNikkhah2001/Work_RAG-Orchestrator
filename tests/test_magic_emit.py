"""Unit tests for deterministic Dana magic-token emission."""

from __future__ import annotations

import json
import re

from work_rag_orchestrator.magic_emit import (
    append_magic_tokens,
    build_magic_tokens,
    collect_magic_payload,
    strip_magic_from_content,
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


def test_person_query_not_forced_cheque_by_answer_prose():
    """Greeting/product lists mentioning چک must not flip حقیقی → cheque."""
    answer = (
        "سلام، من در زمینه‌های گزارش اعتباری، امتیاز و رتبه و بررسی چک‌ها کمک می‌کنم. "
        "امتیاز ۶۱۰ حقیقی در باند سبز است."
    )
    payload = collect_magic_payload(
        {
            "query": "امتیاز ۶۱۰ حقیقی در چه رنگی است؟",
            "supplemental": [_score_ent("person", PERSON_GROUPS)],
            "applied_subspace": "facilities_person",
        },
        answer,
        finish_reason="stop",
    )
    assert payload and payload[0]["name"] == "CREDIT_SCORE_GAUGE"
    assert payload[0]["params"]["model"] == "person"
    assert payload[0]["params"]["min"] == 250


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


def test_shareholders_and_creditors_require_ics_company():
    sh = {
        "similarity": 0.9,
        "content": {"_meta": {"kind": "shareholders"}, "records": []},
    }
    pr = {
        "similarity": 0.9,
        "content": {"_meta": {"kind": "data_providers"}, "records": []},
    }
    # Generic company ask — no ICS context → no widgets
    assert build_magic_tokens(
        {"query": "سهامداران شرکت چه کسانی هستند؟", "supplemental": [sh]},
        finish_reason="stop",
    ) == []
    assert build_magic_tokens(
        {"query": "تامین‌کنندگان دیتا کدام‌اند؟", "supplemental": [pr]},
        finish_reason="stop",
    ) == []
    # Explicit شرکت اعتبارسنجی ایران
    assert any(
        t.startswith("<<<MAGIC:SHAREHOLDERS")
        for t in build_magic_tokens(
            {
                "query": "سهامداران شرکت اعتبارسنجی ایران چه کسانی هستند؟",
                "supplemental": [sh],
            },
            finish_reason="stop",
        )
    )
    assert any(
        t.startswith("<<<MAGIC:CREDITORS")
        for t in build_magic_tokens(
            {
                "query": "تامین‌کنندگان دیتای شرکت اعتبارسنجی ایران کدام‌اند؟",
                "supplemental": [pr],
            },
            finish_reason="stop",
        )
    )


def test_timeline_on_history_query():
    toks = build_magic_tokens(
        {"query": "تاریخچه شرکت اعتبارسنجی ایران را بگو", "supplemental": []},
        finish_reason="stop",
    )
    assert any(t.startswith("<<<MAGIC:COMPANY_TIMELINE") for t in toks)


def test_ics_intro_emits_timeline_not_gauge():
    """Regression: ماده (۵) / امتیاز چک in prose must not invent a cheque gauge."""
    answer = (
        "شرکت اعتبارسنجی ایران (ICS) یک شرکت سهامی خاص است که در آبان‌ماه سال ۱۳۸۵ "
        "بر اساس ماده (۵) قانون تسهیل اعطای تسهیلات بانکی تأسیس شد. "
        "محصولات شامل گزارش و امتیاز اعتباری تسهیلات و همچنین گزارش و امتیاز چک است."
    )
    state = {
        "query": "شرکت اعتبارسنجی ایران چیست؟",
        "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)],
        "_magic_answer_blob": f"شرکت اعتبارسنجی ایران چیست؟\n{answer}",
        "_magic_specific_score": False,
    }
    toks = build_magic_tokens(state, finish_reason="stop")
    names = [re.match(r"<<<MAGIC:([A-Z_]+)", t).group(1) for t in toks]
    assert names == ["COMPANY_TIMELINE"]

    payload = collect_magic_payload(
        {
            "query": "شرکت اعتبارسنجی ایران چیست؟",
            "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)],
        },
        answer,
        finish_reason="stop",
    )
    assert [p["name"] for p in payload] == ["COMPANY_TIMELINE"]


def test_law_article_digit_is_not_cheque_score():
    from work_rag_orchestrator.magic_emit import _blob_has_cheque_numeric, _extract_score

    blob = "بر اساس ماده (۵) قانون تسهیل اعطای تسهیلات بانکی تأسیس شد."
    assert _blob_has_cheque_numeric(blob) is False
    assert _extract_score(blob, _color_groups_to_bands(CHEQUE_GROUPS), 1, 100) is None


def test_precheck_skips_generic_and_provider_counts():
    from work_rag_orchestrator.magic_judge import magic_precheck

    assert magic_precheck("اعتبارسنجی چیست؟", "stop") == frozenset()
    assert magic_precheck("امتیاز من چند است؟", "stop") == frozenset()
    assert "timeline" in magic_precheck("شرکت اعتبارسنجی ایران چیست؟", "stop")
    assert "creditors" in magic_precheck(
        "تأمین‌کنندگان داده شرکت اعتبارسنجی ایران چه نهادهایی هستند؟", "stop"
    )
    assert magic_precheck("رتبه سبز یعنی چه؟", "stop") == frozenset()  # ambiguous colour
    assert "score" in magic_precheck("امتیاز چک ۸۰ در چه رنگی است؟", "stop")
    # High-recall ICS gates
    assert "shareholders" in magic_precheck("سهامداران شرکت اعتبارسنجی ایران", "stop")
    assert "timeline" in magic_precheck("فعالیت شرکت اعتبارسنجی ایران چیست؟", "stop")


def test_magic_judge_prompt_from_registry_not_hardcoded():
    from work_rag_orchestrator.magic_judge import build_magic_judge_prompt

    p = build_magic_judge_prompt("امتیاز چک ۸۰", "", frozenset({"score"}))
    assert "CREDIT_SCORE_GAUGE" in p or "داور" in p or "ویجت" in p
    assert "{query}" not in p  # formatted
    assert "۸۰" in p or "80" in p


def test_ambiguous_colour_no_gauge_on_stop():
    payload = collect_magic_payload(
        {"query": "رتبه سبز یعنی چه؟", "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)]},
        "رتبه سبز در چک بازه ۷۰ تا ۸۵ است.",
        finish_reason="stop",
    )
    assert payload == []


def test_provider_answer_digits_no_gauge():
    payload = collect_magic_payload(
        {
            "query": "شرکت اعتبارسنجی ایران داده‌ها را از چه نهادهایی می‌گیرد؟",
            "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)],
        },
        "۶۸ تأمین‌کننده فعال شامل ۲۸ بانک و ۵ لیزینگ.",
        finish_reason="stop",
    )
    assert [p["name"] for p in payload] == ["CREDITORS"]


def test_company_name_and_poisoned_answer_never_gauge():
    """Judge/precheck must key off the query; answer ماده۵/سبز/چک must not invent a gauge."""
    poison = (
        "شرکت در سال ۱۳۸۵ بر اساس ماده (۵) تأسیس شد. "
        "محصولات شامل امتیاز چک و رتبه سبز است. ۶۸ تأمین‌کننده و رنگ پرچم آبی."
    )
    for q in (
        "شرکت اعتبارسنجی ایران چیست؟",
        "نام شرکت اعتبارسنجی ایران چیست؟",
        "شرکت امتیازدهی اعتباری ایران را معرفی کن",
        "ICS چیست؟",
    ):
        payload = collect_magic_payload(
            {"query": q, "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)]},
            poison,
            finish_reason="stop",
        )
        names = [p["name"] for p in payload]
        assert "CREDIT_SCORE_GAUGE" not in names and "CREDIT_SCORE_BANDS" not in names, (q, names)
        assert names == ["COMPANY_TIMELINE"], (q, names)


def test_clarify_emits_cards():
    toks = build_magic_tokens(
        {
            "query": "وضعیت اعتباری چطور است",
            "needs_clarification": True,
            "clarifying_question": "آیا منظور شما اشخاص حقیقی است یا اشخاص حقوقی؟",
            "available_subspaces": ["گزارش چک", "تسهیلات افراد", "تسهیلات شرکت‌ها"],
        },
        finish_reason="clarify",
    )
    assert len(toks) == 1
    assert toks[0].startswith("<<<MAGIC:CLARIFY_OPTIONS")
    p = _parse_params(toks[0])
    ids = [i["id"] for i in p["items"]]
    # Cards mirror the clarifying fork — not the full available_subspaces dump.
    assert ids == ["person", "legal"]


def test_clarify_cards_match_cheque_facilities_fork():
    toks = build_magic_tokens(
        {
            "query": "رتبه سبز یعنی چه؟",
            "needs_clarification": True,
            "clarifying_question": "آیا منظورتان رتبه رنگی گزارش اعتبارسنجی چک است یا رتبه تسهیلات؟",
        },
        finish_reason="clarify",
    )
    ids = [i["id"] for i in _parse_params(toks[0])["items"]]
    assert ids == ["person", "cheque"] or ids == ["cheque", "person"]


def test_append_keeps_body_clean_and_stashes_payload():
    state = {
        "query": "تاریخچه شرکت اعتبارسنجی ایران",
        "supplemental": [],
    }
    once = append_magic_tokens("متن پاسخ", state, finish_reason="stop")
    twice = append_magic_tokens(once, state, finish_reason="stop")
    assert "<<<MAGIC:" not in once
    assert "<<<MAGIC:" not in twice
    assert once == "متن پاسخ"
    assert twice == "متن پاسخ"
    assert state.get("magic_payload")
    assert state["magic_payload"][0]["name"] == "COMPANY_TIMELINE"


def test_collect_magic_payload_posthoc():
    state = {
        "query": "امتیاز ۶۱۰ حقیقی",
        "supplemental": [_score_ent("person", PERSON_GROUPS)],
        "applied_subspace": "facilities_person",
    }
    payload = collect_magic_payload(state, "امتیاز شما ۶۱۰ است", finish_reason="stop")
    assert payload and payload[0]["name"] == "CREDIT_SCORE_GAUGE"
    assert payload[0]["params"]["score"] == 610
    assert payload[0]["params"]["model"] == "person"
    assert "codes" in payload[0]["params"]["bands"][0] or payload[0]["params"].get("codes") is not None


def test_generic_credit_ask_emits_no_gauge():
    payload = collect_magic_payload(
        {
            "query": "اعتبارسنجی چیست؟",
            "supplemental": [_score_ent("person", PERSON_GROUPS)],
        },
        "اعتبارسنجی فرایند ارزیابی ریسک اعتباری است.",
        finish_reason="stop",
    )
    assert payload == []


def test_strip_magic_from_content():
    raw = "پاسخ\n\n<<<MAGIC:CREDIT_SCORE_GAUGE {\"score\":10}>>>\n"
    assert "<<<MAGIC:" not in strip_magic_from_content(raw)
    assert "پاسخ" in strip_magic_from_content(raw)


def test_linear_variant():
    state = {
        "query": "نمایش خطی امتیاز چک ۸۰",
        "supplemental": [_score_ent("cheque", CHEQUE_GROUPS)],
        "applied_subspace": "cheque",
    }
    p = _parse_params(build_magic_tokens(state, finish_reason="stop")[0])
    assert p["variant"] == "linear"
    assert p["model"] == "cheque"


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
