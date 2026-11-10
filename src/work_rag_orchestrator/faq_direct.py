"""FAQ direct-answer shortlist — bypass retrieval for curated questions.

When a user query closely matches one of these FAQs, the RAG pipeline
skips KB retrieval and answers directly from the curated content via LLM.
This keeps official/support answers deterministic and avoids retrieval noise.

Matching is intentionally fuzzy: exact → substring → token containment →
difflib → embedding cosine. The user should not need character-perfect phrasing.
"""

from __future__ import annotations

import logging
import difflib

log = logging.getLogger(__name__)

# Curated Q → A pairs live in KB canned pack (`faq_shortlist`).


def _faq_pairs() -> dict[str, str]:
    from .content_registry import faq_shortlist

    return faq_shortlist()


class _FaqProxy(dict):
    """Lazy dict proxy so existing ``FAQ_SHORTLIST`` call sites keep working."""

    def _data(self) -> dict[str, str]:
        return _faq_pairs()

    def __iter__(self):
        return iter(self._data())

    def items(self):
        return self._data().items()

    def keys(self):
        return self._data().keys()

    def values(self):
        return self._data().values()

    def __len__(self):
        return len(self._data())

    def __contains__(self, key):
        return key in self._data()

    def __getitem__(self, key):
        return self._data()[key]

    def get(self, key, default=None):
        return self._data().get(key, default)


FAQ_SHORTLIST = _FaqProxy()


def _normalize(text: str) -> str:
    """Persian-aware normalization for FAQ matching (mirrors retrieve.py)."""
    t = (text or "").strip().lower()
    # unify ZWNJ, yeh/kaf variants, and common compounds
    t = t.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
    # handle compound سوءاثر variants without space
    t = t.replace("سوءاثر", "سوء اثر").replace("سواثر", "سوء اثر")
    # informal shortcuts & verb variations
    t = t.replace("چیکار", "چه کار").replace("چطور", "چگونه").replace("کی", "چه زمانی")
    t = t.replace("گرفتم", "گرفته").replace("گرفتیم", "گرفته").replace("گرفتید", "گرفته")
    t = t.replace("جدید ", "جدیدا ").replace("جدیداا", "جدیدا")
    t = t.replace("میشه", "می شود").replace("میشود", "می شود").replace("میشم", "می شود")
    t = t.replace("چی میشه", "چه می شود").replace("چی می شود", "چه می شود")
    # strip common punctuation (including Persian ؟ ،)
    for ch in "؟?!.,،؛:؛\"'«»()[]{}…-–—\n\r\t":
        t = t.replace(ch, " ")
    # collapse whitespace
    t = " ".join(t.split())
    return t


import math
from collections import Counter

_NORMALIZED_FAQ: dict[str, dict[str, str]] = {}
_N = 0
_DF: Counter = Counter()
_IDF: dict[str, float] = {}
_FAQ_INDEX_BUILT = False


def _ensure_faq_index() -> None:
    """Build normalized FAQ + IDF maps from the registry shortlist (once)."""
    global _NORMALIZED_FAQ, _N, _DF, _IDF, _FAQ_INDEX_BUILT
    if _FAQ_INDEX_BUILT and _NORMALIZED_FAQ:
        return
    _NORMALIZED_FAQ = {}
    for q, a in _faq_pairs().items():
        _NORMALIZED_FAQ[_normalize(q)] = {"answer": a, "question": q}
    _N = max(len(_NORMALIZED_FAQ), 1)
    _DF = Counter()
    for norm_q in _NORMALIZED_FAQ:
        for tok in set(norm_q.split()):
            if len(tok) > 1:
                _DF[tok] += 1
    _IDF = {t: math.log(_N / df) for t, df in _DF.items()}
    _FAQ_INDEX_BUILT = True


# --- Optional embedding index for fuzzy semantic matching ---
_EMBED_MODEL = None
_EMBED_DIM = 384
_FAQ_EMBEDDINGS: dict[str, list[float]] | None = None  # norm_q -> vector
_EMBED_THRESHOLD = 0.72  # cosine similarity threshold for FAQ hit

def _try_load_embed_model():
    _ensure_faq_index()
    global _EMBED_MODEL, _FAQ_EMBEDDINGS
    if _EMBED_MODEL is not None or _FAQ_EMBEDDINGS is not None:
        return
    try:
        # offline-prep stores model with _ instead of /
        import os
        from sentence_transformers import SentenceTransformer
        candidates = [
            os.getenv("FAQ_EMBED_MODEL_PATH", "/tmp/hf_clean/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2"),
            "/splunk-data/v1/Work_RAG-Server-Setup/offline-prep/models/huggingface/sentence-transformers_all-MiniLM-L6-v2",
        ]
        model_path = next((p for p in candidates if os.path.isdir(p)), None)
        if model_path is None:
            # fall back to HF cache name (will try hub offline)
            model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        _EMBED_MODEL = SentenceTransformer(model_path, device="cpu", trust_remote_code=False)
        # pre-compute FAQ embeddings (normalized questions)
        questions = list(_NORMALIZED_FAQ.keys())
        vectors = _EMBED_MODEL.encode(questions, normalize_embeddings=True, show_progress_bar=False)
        _FAQ_EMBEDDINGS = {q: v.tolist() if hasattr(v, "tolist") else list(v) for q, v in zip(questions, vectors)}
        log.info("FAQ embed index built: %d entries dim=%d from %s", len(_FAQ_EMBEDDINGS), _EMBED_DIM, model_path)
    except Exception as e:
        log.warning("FAQ embedding index not available (%s) — falling back to token/difflib only", e)
        _EMBED_MODEL = None
        _FAQ_EMBEDDINGS = None


# Lazily try once at import — non-blocking; if it fails we still have token/difflib
try:
    _try_load_embed_model()
except Exception:
    pass


def _cosine(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    # vectors are L2-normalized, so dot is cosine
    return float(dot)


# Also index without trailing question particles for looser match
# e.g. "مدیرعامل شرکت اعتبارسنجی ایران کیست" vs "مدیرعامل ... کیست؟"
def find_direct_answer(query: str) -> str | None:
    """Return canned answer if query matches a FAQ entry, else None.

    Matching strategy (in order, fuzzy):
      1. exact normalized match
      2. substring containment (len>8, ratio>=0.5)
      3. token-set containment (>=60% of smaller set, >=3 tokens)
      4. difflib SequenceMatcher ratio >=0.72
      5. embedding cosine >=0.72 (paraphrase-multilingual-MiniLM-L12-v2)
    """
    _ensure_faq_index()
    nq = _normalize(query)
    if not nq:
        return None

    # 1) exact normalized match
    if nq in _NORMALIZED_FAQ:
        return _NORMALIZED_FAQ[nq]["answer"]

    # 2) substring containment — balanced: either string contains the other
    for norm_q, payload in _NORMALIZED_FAQ.items():
        if len(norm_q) > 8 and len(nq) > 8 and (norm_q in nq or nq in norm_q):
            longer = max(len(norm_q), len(nq))
            shorter = min(len(norm_q), len(nq))
            # stricter ratio to avoid generic "اعتبارسنجی چیست" matching longer FAQ
            if shorter / longer >= 0.75 and abs(len(norm_q) - len(nq)) <= 15:
                return payload["answer"]

    # 3) embedding semantic similarity — handles true paraphrases, before token to pick correct FAQ
    # e.g. "اطلاعات گزارش اعتباری کی به روز میشه؟" vs "اطلاعات موجود در گزارش اعتباری هر چند وقت یک‌بار به روزرسانی می‌شوند؟"
    # Guard: very short/generic queries (<20 chars or <3 tokens) skip embedding to avoid false positives
    try:
        if len(nq) >= 20:
            q_tokens_tmp = set(t for t in nq.split() if len(t) > 1)
            if len(q_tokens_tmp) >= 3:
                if _FAQ_EMBEDDINGS is None:
                    _try_load_embed_model()
                if _EMBED_MODEL is not None and _FAQ_EMBEDDINGS is not None:
                    q_vec = _EMBED_MODEL.encode([nq], normalize_embeddings=True, show_progress_bar=False)[0]
                    q_list = q_vec.tolist() if hasattr(q_vec, "tolist") else list(q_vec)
                    best_score = -1.0
                    best_emb_payload = None
                    for norm_q, payload in _NORMALIZED_FAQ.items():
                        vec = _FAQ_EMBEDDINGS.get(norm_q)
                        if vec is None:
                            continue
                        score = _cosine(q_list, vec)
                        if score > best_score:
                            best_score = score
                            best_emb_payload = payload
                    # higher threshold to avoid generic hits (0.78)
                    if best_score >= 0.85 and best_emb_payload is not None:
                        log.info("FAQ embedding hit score=%.3f for query=%r -> %r", best_score, query[:60], best_emb_payload["question"][:40])
                        return best_emb_payload["answer"]
    except Exception as e:
        log.warning("FAQ embedding check failed: %s", e)

    # 4) token-set containment with IDF weighting — pick best FAQ
    q_tokens = set(t for t in nq.split() if len(t) > 1)
    best_tok_score = 0
    best_tok_payload = None
    for norm_q, payload in _NORMALIZED_FAQ.items():
        faq_tokens = set(t for t in norm_q.split() if len(t) > 1)
        if not faq_tokens or not q_tokens:
            continue
        inter = q_tokens & faq_tokens
        if len(inter) < 2:
            continue
        # guard: generic 2-token common queries
        if len(q_tokens) <= 2 and len(inter) <= 2:
            if all(_DF.get(t, 0) > _N * 0.2 for t in inter):
                continue
        # allow 2-token only if one is rare
        if len(inter) == 2 and not any(_DF.get(t, 0) <= 3 for t in inter):
            if len(q_tokens) <= 3:
                continue
        faq_idf_sum = sum(_IDF.get(t, 1.0) for t in faq_tokens)
        inter_idf_sum = sum(_IDF.get(t, 1.0) for t in inter)
        idf_ratio = inter_idf_sum / faq_idf_sum if faq_idf_sum else 0
        smaller = min(len(q_tokens), len(faq_tokens))
        cnt_ratio = len(inter) / smaller if smaller else 0
        jaccard = len(inter) / len(q_tokens | faq_tokens) if (q_tokens | faq_tokens) else 0
        score = max(idf_ratio*1.2, cnt_ratio, jaccard*1.1)
        # boost rare overlap
        if any(_DF.get(t, 0) <= 2 for t in inter):
            score += 0.15
        if len(faq_tokens) <= 4 and len(inter) >= len(faq_tokens) - 1:
            score = max(score, 0.75)
        if score > best_tok_score:
            best_tok_score = score
            best_tok_payload = payload
    if best_tok_payload is not None and best_tok_score >= 0.60:
        return best_tok_payload["answer"]

    # 5) difflib fuzzy string ratio — stricter to avoid generic hits
    best_ratio = 0.0
    best_payload = None
    for norm_q, payload in _NORMALIZED_FAQ.items():
        r = difflib.SequenceMatcher(None, nq, norm_q).ratio()
        if r > best_ratio:
            best_ratio = r
            best_payload = payload
    if best_ratio >= 0.80 and best_payload is not None:
        if len(nq) >= 15 and len(nq.split()) >= 3:
            return best_payload["answer"]

    return None


def is_faq_query(query: str) -> bool:
    """Convenience: does this query belong to the FAQ shortlist?"""
    return find_direct_answer(query) is not None


def list_faq_questions() -> list[str]:
    """Return canonical FAQ question list (for debug/dashboard)."""
    _ensure_faq_index()
    return list(FAQ_SHORTLIST.keys())
