"""Compound-phrase lexical helpers (e.g. bank guarantee letter vs personal guarantor)."""

from __future__ import annotations

import re
from typing import Any

# ضمانت‌نامه بانکی vs ضامن / «ضمانت کردن» — BM25 overlaps on bare «ضمانت».
_GUARANTEE_LETTER_RE = re.compile(r"ضمانت\s*نامه", re.IGNORECASE)
_GUARANTOR_SKEW_MARKERS = (
    "ضامن",
    "ضمانت کردن",
    "ضمانت برای دیگران",
    "ضمانت کرده",
    "بدون ضامن",
)

_PHRASE_BOOST = 8.0
_GUARANTOR_PENALTY = 6.0


def normalize_fa_text(text: str) -> str:
    """Unify ZWNJ/space and Arabic-letter variants for substring checks."""
    t = (text or "").replace("ي", "ی").replace("ك", "ک")
    t = t.replace("\u200c", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _compact_no_space(text: str) -> str:
    return normalize_fa_text(text).replace(" ", "")


def query_wants_bank_guarantee_letter(query: str) -> bool:
    """True when the user asks about bank guarantee letters, not personal guarantors."""
    if not query:
        return False
    n = normalize_fa_text(query)
    if _GUARANTEE_LETTER_RE.search(n):
        return True
    compact = _compact_no_space(query)
    return "ضمانتنامه" in compact


def chunk_has_guarantee_letter(text: str) -> bool:
    n = normalize_fa_text(text)
    if _GUARANTEE_LETTER_RE.search(n):
        return True
    return "ضمانتنامه" in _compact_no_space(text)


def chunk_is_guarantor_skewed(text: str) -> bool:
    """Personal-guarantor FAQ signals without bank guarantee-letter wording."""
    if chunk_has_guarantee_letter(text):
        return False
    n = normalize_fa_text(text)
    return any(m in n for m in _GUARANTOR_SKEW_MARKERS)


def compound_phrase_score(query: str, chunk: dict[str, Any]) -> float:
    """Additive score bump for retrieve reranking (higher = prefer this chunk)."""
    if not query_wants_bank_guarantee_letter(query):
        return 0.0
    blob = " ".join(
        str(chunk.get(k) or "")
        for k in ("content", "heading", "title")
    )
    score = 0.0
    if chunk_has_guarantee_letter(blob):
        score += _PHRASE_BOOST
    if chunk_is_guarantor_skewed(blob):
        score -= _GUARANTOR_PENALTY
    return score


def _base_chunk_score(chunk: dict[str, Any]) -> float:
    best = 0.0
    for key in ("rerank_score", "hybrid_score", "score", "fused_rrf_score"):
        v = chunk.get(key)
        if isinstance(v, (int, float)):
            best = max(best, float(v))
    return best


def rerank_chunks_by_compound_phrases(chunks: list[dict[str, Any]] | None, query: str) -> list[dict[str, Any]]:
    """Stable rerank: compound-phrase boost on top of hybrid/rerank scores."""
    if not chunks or not query:
        return list(chunks or [])
    if not query_wants_bank_guarantee_letter(query):
        return list(chunks)

    def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[float, float, int]:
        idx, ch = item
        phrase = compound_phrase_score(query, ch)
        base = _base_chunk_score(ch)
        return (phrase + base, base, -idx)

    indexed = list(enumerate(chunks))
    indexed.sort(key=sort_key, reverse=True)
    return [ch for _, ch in indexed]


def anchor_guarantee_letter_in_search_query(user_thread_text: str, search_query: str) -> str:
    """If any user turn mentioned ضمانت‌نامه, keep the bigram in KB search_query.

    Rewrites sometimes drop «نامه», leaving bare «ضمانت» which BM25 confuses
    with personal guarantor (ضامن) FAQs under facilities filters.
    """
    if not query_wants_bank_guarantee_letter(user_thread_text):
        return search_query
    if query_wants_bank_guarantee_letter(search_query):
        return search_query
    sq = normalize_fa_text(search_query)
    # Prepend anchored bigram; strip a lone leading «ضمانت» token if present.
    sq = re.sub(r"^ضمانت\s+", "", sq, count=1)
    return f"ضمانت نامه {sq}".strip()


def lexical_evidence_requires_guarantee_letter(query: str) -> bool:
    """When True, generic «ضمانت» overlap must not count as lexical evidence."""
    return query_wants_bank_guarantee_letter(query)


def has_guarantee_letter_lexical_evidence(query: str, chunks: list) -> bool:
    """Any chunk contains the bank guarantee-letter phrase (not bare ضمانت)."""
    if not chunks:
        return False
    for c in chunks[:10]:
        blob = str(c.get("content") or "")
        if chunk_has_guarantee_letter(blob):
            return True
    return False