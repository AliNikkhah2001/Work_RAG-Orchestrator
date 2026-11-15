"""Unit tests for bank guarantee-letter vs personal guarantor disambiguation."""

from __future__ import annotations

from work_rag_orchestrator.lexical_phrase_ranking import (
    anchor_guarantee_letter_in_search_query,
    chunk_has_guarantee_letter,
    chunk_is_guarantor_skewed,
    compound_phrase_score,
    query_wants_bank_guarantee_letter,
    rerank_chunks_by_compound_phrases,
)


def test_query_detects_guarantee_letter_zwnj_and_space():
    assert query_wants_bank_guarantee_letter("ضمانت‌نامه بانکی چه ربطی دارد؟")
    assert query_wants_bank_guarantee_letter("ضمانت نامه بانکی")
    assert not query_wants_bank_guarantee_letter("ضمانت کردن برای دیگران")


def test_anchor_restores_dropped_nameh():
    thread = "ضمانت‌نامه بانکی چه ربطی به اعتبارسنجی دارد؟ حقوقی"
    degraded = "ضمانت بانکی اعتبارسنجی"
    anchored = anchor_guarantee_letter_in_search_query(thread, degraded)
    assert "ضمانت نامه" in anchored
    assert anchored.startswith("ضمانت نامه")


def test_chunk_signals():
    letter = "سوال: اگر شرکت ضمانت‌نامه بگیرد..."
    guarantor = "سوال: اگر شرکت وامی را ضمانت کرده باشد..."
    assert chunk_has_guarantee_letter(letter)
    assert not chunk_is_guarantor_skewed(letter)
    assert chunk_is_guarantor_skewed(guarantor)
    assert not chunk_has_guarantee_letter(guarantor)


def test_rerank_prefers_guarantee_letter_faq():
    q = "ضمانت‌نامه بانکی اعتبارسنجی"
    guarantor = {
        "content": "شرکت وامی را به نام شخص دیگری ضمانت کرده",
        "hybrid_score": 0.9,
    }
    letter = {
        "content": "ضمانت‌نامه‌های صادر شده در امتیاز اعتباری لحاظ می‌شوند",
        "hybrid_score": 0.2,
    }
    assert compound_phrase_score(q, letter) > compound_phrase_score(q, guarantor)
    ranked = rerank_chunks_by_compound_phrases([guarantor, letter], q)
    assert chunk_has_guarantee_letter(ranked[0]["content"])