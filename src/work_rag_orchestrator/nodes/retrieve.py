"""Retrieve node - calls KB search API."""

from __future__ import annotations

import logging
import time
from ..state import RAGState, record_timing
from ..config import get_settings
from ..subspaces import SUBSPACES
from ..clients.knowledgebase import KnowledgebaseClient
from ..clients.faq_verifier import verify_faq_answer
from ..rewrite import rewrite_query, last_exchanges_text
from ..faq_direct import find_direct_answer

log = logging.getLogger(__name__)

# Messages that need no retrieval: exact match after normalization.
# Checked against the RAW last user message BEFORE history rewrite, so a bare
# "سلام" can never be rewritten into an older question and RAG-fabricated.
_GREETING = {
    "سلام", "سلام وقت بخیر", "سلام صبح بخیر", "سلام عصر بخیر", "سلام شب بخیر",
    "درود", "درود بر شما", "هی", "هیلو", "hello", "hi", "hey", "hey there",
    "صبح بخیر", "عصر بخیر", "شب بخیر", "روز بخیر", "وقت بخیر",
}
_FAREWELL = {"خداحافظ", "خدا حافظ", "بای", "bye", "goodbye", "فعلا", "فعلاً"}
_THANKS = {
    "ممنون", "ممنونم", "مرسی", "تشکر", "متشکرم", "متشکرم", "تشکر میکنم",
    "دست شما درد نکند", "دستت درد نکند", "عالی", "باشه", "اوکی", "ok", "thanks", "thank you",
}


def detect_greeting_kind(text: str) -> str | None:
    """Return 'greeting'/'farewell'/'thanks' for standalone smalltalk, else None."""
    import re

    t = (text or "").strip().lower()
    # Strip trailing ASCII dots (e.g. "hello..." -> "hello", "سلام..." -> "سلام") before generic punctuation removal
    t = re.sub(r"\.+$", "", t).strip()
    for ch in "؟?!.,،؛:؛\"'«»()…-":
        t = t.replace(ch, "")
    t = re.sub(r"\.+$", "", t).strip()
    t = " ".join(t.replace("‌", " ").split())
    if not t:
        return None
    if t in _FAREWELL:
        return "farewell"
    if t in _THANKS:
        return "thanks"
    if t in _GREETING:
        return "greeting"
    return None


def _extract_subspace(hier: list[str] | str | None) -> str:
    """Map folder_hierarchy to subspace label (F4)."""
    s = "/".join(hier) if isinstance(hier, list) else str(hier or "")
    if "اشخاص حقوقی" in s:
        return "facilities_legal"
    elif "گزارش اعتباری چک" in s:
        return "cheque"
    elif "پایگاه دانش عمومی" in s:
        return "general"
    else:
        return "facilities_person"


# --- Dynamic retrieval: one re-retrieval then not-enough-context + DLQ ---
_DLQ_PATH = "/tmp/dlq_retrieval.jsonl"


def _compute_top_score(chunks: list | None) -> float:
    """Best rerank/hybrid/score across chunks, 0.0 if none.

    T6 fix: MAX across all keys (don't break after first), so hybrid 0.03 is
    not hidden behind rerank 0.0.
    """
    best = 0.0
    for c in (chunks or []):
        for k in ("rerank_score", "hybrid_score", "score"):
            v = c.get(k)
            if isinstance(v, (int, float)):
                if float(v) > best:
                    best = float(v)
    return best


def _has_lexical_evidence_retrieve(query: str, chunks: list | None, applied_subspace: str | None) -> bool:
    """T6 lexical proxy for ctx_rel (retrieve side) — same logic as format_response."""
    if not chunks:
        return False
    q = (query or "").strip()
    if not q:
        return False
    subspace_kw = {
        "cheque": ["چک"],
        "facilities_person": ["تسهیلات", "وام", "قسط", "حقیقی"],
        "facilities_legal": ["حقوقی", "شرکت"],
        "general": ["معرفی", "مدیر", "شرکت", "عمومی", "عام"],
    }
    keywords: set[str] = set()
    s = (applied_subspace or "").strip().lower()
    if "چک" in (applied_subspace or "") or s == "cheque":
        keywords.update(subspace_kw["cheque"])
    elif "حقیقی" in (applied_subspace or "") or s == "facilities_person":
        keywords.update(subspace_kw["facilities_person"])
    elif "حقوقی" in (applied_subspace or "") or s == "facilities_legal":
        keywords.update(subspace_kw["facilities_legal"])
    elif "عمومی" in (applied_subspace or "") or "پایگاه دانش عمومی" in (applied_subspace or "") or s == "general":
        keywords.update(subspace_kw["general"])
    import re as _re2
    q_norm = _re2.sub(r"[؟?!.,،؛:\"'«»()…\-\.]+", " ", q)
    for tok in q_norm.split():
        t = tok.strip()
        if len(t) >= 2 and t not in {"در", "با", "از", "به", "که", "را", "یا", "هم", "این", "آن", "برای", "آیا", "چقدر", "چطور", "چگونه", "چیست", "است"}:
            keywords.add(t)
    if not keywords:
        return False
    corpus = " ".join(str(c.get("content") or "") for c in (chunks or [])[:10])
    if not corpus:
        return False
    for kw in keywords:
        if kw and kw in corpus:
            return True
        kw2 = kw.replace("‌", " ").strip()
        if kw2 and kw2 != kw and kw2 in corpus:
            return True
    return False


def _get_ev_threshold(applied_subspace: str | None = None) -> float:
    try:
        from ..config import get_settings as _gs
        return float(_gs().get_evidence_threshold(applied_subspace))
    except Exception:
        # fallback hardcoded per-subspace defaults (T3)
        s = (applied_subspace or "")
        low = s.lower() if isinstance(s, str) else ""
        if "چک" in s or low == "cheque":
            return 0.05
        if "حقیقی" in s or low == "facilities_person":
            return 0.10
        return 0.15


def _subspace_keyword_boost(subspace_or_filter: str | None) -> float:
    """D: BM25 keyword_boost for a scoped (filtered) retrieval call.

    Accepts either the canonical subspace label (cheque, facilities_person,
    facilities_legal, general) or the Persian filter_path itself (چک،
    حقیقی، حقوقی، عمومی substrings). Unknown or missing values fall back
    to 3.0 — the KB server default — so unscoped calls behave exactly as
    before.
    """
    default = 3.0
    try:
        boost_map = get_settings().subspace_keyword_boost or {}
    except Exception:
        boost_map = {}
    s = (subspace_or_filter or "").strip()
    if not s:
        return default
    if s in boost_map:
        try:
            return float(boost_map[s])
        except (TypeError, ValueError):
            return default
    low = s.lower()
    if "چک" in s or low == "cheque":
        key = "cheque"
    elif "حقیقی" in s or low == "facilities_person":
        key = "facilities_person"
    elif "حقوقی" in s or low == "facilities_legal":
        key = "facilities_legal"
    elif "عمومی" in s or low == "general":
        key = "general"
    else:
        return default
    try:
        return float(boost_map.get(key, default))
    except (TypeError, ValueError):
        return default


def _save_dlq_sample(
    state: dict,
    request_id: str,
    query: str,
    search_query: str,
    top_score: float,
    threshold: float,
    stage: str,
) -> None:
    """Clear chunks for refusal, mark DLQ, trace and append JSONL."""
    # one-retry budget: mark attempt
    try:
        state["retrieve_attempts"] = int(state.get("retrieve_attempts") or 0) + 1
    except Exception:
        state["retrieve_attempts"] = 1
    state["retrieved_chunks"] = []
    state["dlq_saved"] = True
    # trace_span for observability
    try:
        from ..tracing import trace_span as _tsp

        _tsp(
            request_id,
            "retrieve-dlq",
            span_input={"query": query, "search_query": search_query, "top_score": top_score, "threshold": threshold, "stage": stage, "retrieve_attempts": state.get("retrieve_attempts")},
            span_output={"dlq": True, "query": query, "cleared": True},
        )
    except Exception:
        pass
    # JSONL file for offline DLQ review
    try:
        import json as _json
        import time as _time

        rec = {
            "ts": _time.time(),
            "request_id": request_id,
            "query": query,
            "search_query": search_query,
            "top_score": top_score,
            "threshold": threshold,
            "stage": stage,
            "dlq": True,
            "retrieve_attempts": state.get("retrieve_attempts"),
        }
        with open(_DLQ_PATH, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass
    log.info("DLQ saved for request %s stage=%s top=%.4f < %.3f query=%r", request_id, stage, top_score, threshold, (query or search_query)[:80])


# ---------------------------------------------------------------------------
# T1: Filtered → whole-KB fallback helpers
# ---------------------------------------------------------------------------

def _t1_item_dict(c, default_source: str) -> dict:
    """Normalize a KB result item to the internal chunk dict (shared by fallback)."""
    d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
    get = d.get
    return {
        "chunk_id": get("chunk_id"),
        "document_id": get("document_id"),
        "title": get("title"),
        "heading": get("heading"),
        "content": get("content"),
        "score": get("score"),
        "source": get("source") or default_source,
        "rank_rrf": get("rank_rrf"),
        "rank_ce": get("rank_ce"),
        "hybrid_score": get("hybrid_score"),
        "rerank_score": get("rerank_score"),
        "folder_hierarchy": get("folder_hierarchy"),
    }


def _t1_merge_rrf_ce(rrf_chunks, ce_chunks) -> list[dict]:
    """Union top-10 RRF + top-10 CE, deduped by chunk_id, ordered both>ce>rrf (T1 reuse)."""
    merged: dict[str, dict] = {}
    order: list[str] = []
    for c in rrf_chunks:
        d = _t1_item_dict(c, "rrf")
        key = d.get("chunk_id") or f"__rrf_{len(order)}"
        if key not in merged:
            merged[key] = d
            order.append(key)
    for c in ce_chunks:
        d = _t1_item_dict(c, "ce")
        key = d.get("chunk_id") or f"__ce_{len(order)}"
        if key in merged:
            existing = merged[key]
            existing["source"] = "both"
            if existing.get("rank_ce") is None:
                existing["rank_ce"] = d.get("rank_ce")
            if existing.get("rerank_score") is None:
                existing["rerank_score"] = d.get("rerank_score")
            if existing.get("hybrid_score") is None:
                existing["hybrid_score"] = d.get("hybrid_score")
            existing["score"] = (
                existing.get("rerank_score")
                if existing.get("rerank_score") is not None
                else d.get("score", existing.get("score"))
            )
            for k in ("document_id", "title", "heading", "content", "folder_hierarchy"):
                if not existing.get(k) and d.get(k):
                    existing[k] = d[k]
        else:
            merged[key] = d
            order.append(key)

    def _best_rank(d: dict) -> int:
        ranks = [r for r in (d.get("rank_rrf"), d.get("rank_ce")) if isinstance(r, int)]
        return min(ranks) if ranks else 10**9

    both = sorted((merged[k] for k in order if merged[k].get("source") == "both"), key=_best_rank)
    ce_only = sorted((merged[k] for k in order if merged[k].get("source") == "ce"), key=lambda d: d.get("rank_ce") if isinstance(d.get("rank_ce"), int) else 10**9)
    rrf_only = sorted((merged[k] for k in order if merged[k].get("source") == "rrf"), key=lambda d: d.get("rank_rrf") if isinstance(d.get("rank_rrf"), int) else 10**9)
    return (both + ce_only + rrf_only)[:20]


def _fuse_subspace_rrf(filtered: list, whole: list, filtered_weight: float = 1.3) -> list:
    """A: Reciprocal-rank fusion of the filtered (subspace) and whole-KB legs.

    score(d) = 1 / (60 + rank_in_list) * weight, summed over every leg the
    chunk appears in.  The filtered leg carries ``filtered_weight`` (1.3x) so:
      - chunks present in BOTH lists are boosted to the top,
      - chunks only in the filtered subspace keep priority over most
        whole-KB noise,
      - whole-KB-only chunks (the recall the filter was killing) still
        enter the tail instead of being hard-replaced or dropped.

    Deduped by chunk_id; missing display fields/ranks are merged from the
    other leg, and absolute scores (score/rerank_score/hybrid_score) keep
    the max so downstream evidence-threshold checks see the best leg.
    Each output item gets ``fused_rrf_score`` for audit/debug visibility.
    """
    k = 60.0
    scores: dict = {}
    entries: dict = {}
    for chunks, weight in ((filtered, filtered_weight), (whole, 1.0)):
        for rank, c in enumerate(chunks or [], start=1):
            if not isinstance(c, dict):
                continue
            key = c.get("chunk_id") or f"__fuse_{id(c)}"
            scores[key] = scores.get(key, 0.0) + (weight / (k + rank))
            if key not in entries:
                entries[key] = dict(c)
                continue
            existing = entries[key]
            # Fill missing display fields from the other leg's copy.
            for field in ("document_id", "title", "heading", "content", "folder_hierarchy"):
                if not existing.get(field) and c.get(field):
                    existing[field] = c.get(field)
            # Keep the best absolute score per field (evidence checks read these).
            for field in ("score", "rerank_score", "hybrid_score"):
                a, b = existing.get(field), c.get(field)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    existing[field] = max(a, b)
                elif existing.get(field) is None and isinstance(b, (int, float)):
                    existing[field] = b
            # Fill missing rank fields from the other leg.
            for field in ("rank_rrf", "rank_ce"):
                if existing.get(field) is None and c.get(field) is not None:
                    existing[field] = c.get(field)
    # Sort by fused RRF score desc; ties keep insertion order (filtered leg first).
    out = []
    for kk in sorted(entries, key=lambda key: scores[key], reverse=True):
        item = entries[kk]
        item["fused_rrf_score"] = round(scores[kk], 6)
        out.append(item)
    return out


async def _t1_try_whole_kb_fallback(
    state: dict,
    search_query: str,
    request_id: str,
    top_k: int = 20,
    fallback_reason: str = "",
    active_filter: str | None = None,
) -> bool:
    """
    T1: If current retrieval was filtered and evidence is weak (0 chunks or
    top_score < evidence_min_score), retry once against the whole KB
    (filter_path=None, 1123 chunks).  Single retry only; avoid loop when
    already whole-KB; handle exceptions gracefully keeping original results.

    A: when both the filtered list and the whole-KB list are non-empty, the
    two legs are FUSED via weighted RRF (_fuse_subspace_rrf) instead of a
    hard replace; the audit fallback_reason mentions "fused".

    Returns True if a fallback attempt was made (even if no chunks adopted).
    Records state["fallback_used"], state["fallback_reason"], and
    state["stage_timing_ms"]["retrieve_retry"] plus tracing.
    """
    # Avoid infinite loop: if original probe was already whole-KB, skip.
    if not active_filter:
        # No filtered probe — whole KB was already used.
        return False
    # Single retry guard.
    if state.get("fallback_used"):
        return False
    # Check trigger: 0 chunks or max score < threshold (T3 per-subspace).
    cur_chunks = state.get("retrieved_chunks") or []
    cur_top = _compute_top_score(cur_chunks)
    thr = _get_ev_threshold(active_filter)
    cur_len = len(cur_chunks)
    # Also consider FAQ-verified bypass: don't fallback if FAQ satisfied.
    if state.get("faq_verified"):
        return False
    needs_fallback = (cur_len == 0) or (cur_top < thr)
    if not needs_fallback:
        return False

    # Determine fallback reason string for audit.
    if not fallback_reason:
        if cur_len == 0:
            fallback_reason = f"filtered_empty sub={active_filter}"
        else:
            fallback_reason = f"low_score top={cur_top:.4f} < thr={thr:.3f} sub={active_filter}"

    t_retry = time.monotonic()
    log.info("T1 fallback: filtered (%s) weak (len=%d top=%.4f < %.3f) — retrying whole KB with query=%r", active_filter, cur_len, cur_top, thr, search_query[:80])
    try:
        from ..clients.knowledgebase import KnowledgebaseClient as _KBClient

        # KnowledgebaseClient uses httpx.AsyncClient(trust_env=False) internally,
        # so localhost bypasses corporate Squid proxy (critical for 127.0.0.1).
        async with _KBClient() as fb_client:
            fb_result = await fb_client.retrieve(search_query, top_k, request_id, filter_path=None)
    except Exception as e:
        # Retry failed — keep original, but record fallback attempt for audit.
        elapsed = round((time.monotonic() - t_retry) * 1000, 1)
        state["fallback_used"] = True
        state["fallback_reason"] = f"{fallback_reason} retry_failed: {e}"
        record_timing(state, "retrieve_retry", elapsed)
        log.warning("T1 whole-KB retry failed for request %s (keep original): %s", request_id, e)
        try:
            from ..tracing import trace_span as _tsp

            _tsp(request_id, "retrieve-retry", span_input={"query": search_query, "active_filter": active_filter, "cur_len": cur_len, "cur_top": cur_top, "threshold": thr, "reason": fallback_reason}, span_output={"error": str(e), "fallback_used": True})
        except Exception:
            pass
        return True

    # Merge fallback result using same RRF+CE union logic.
    try:
        if isinstance(fb_result, dict):
            fb_rrf = fb_result.get("rrf", [])
            fb_ce = fb_result.get("ce", [])
            fb_injected = fb_result.get("injected_entities", [])
        else:
            fb_rrf = []
            fb_ce = list(fb_result)
            fb_injected = []
        fb_retrieved = _t1_merge_rrf_ce(fb_rrf, fb_ce)
        # Prepend FAQ if verified (same as main path).
        if state.get("faq_verified") and state.get("direct_faq_answer"):
            fb_retrieved.insert(0, {
                "chunk_id": "__faq_direct__",
                "document_id": "__faq_direct__",
                "title": "FAQ Direct Answer",
                "heading": "FAQ",
                "content": state["direct_faq_answer"],
                "score": 1.0,
                "source": "faq_direct",
                "rank_rrf": 0,
                "rank_ce": 0,
                "hybrid_score": 1.0,
                "rerank_score": None,
            })
        fb_top = _compute_top_score(fb_retrieved)
        fb_len = len(fb_retrieved)
        elapsed = round((time.monotonic() - t_retry) * 1000, 1)
        # A: subspace re-rank — ALWAYS FUSE the filtered leg with the whole-KB
        # leg via weighted RRF when both lists are non-empty (no more hard
        # replace). Chunks in both lists get boosted, chunks only in the
        # filtered subspace keep priority over most whole-KB noise, and
        # whole-KB-only chunks (the recall the filter was killing) enter the
        # tail. Wholesale replace survives only for the cur_len == 0 case.
        state["fallback_used"] = True
        record_timing(state, "retrieve_retry", elapsed)
        fused_now = False
        if cur_len > 0 and fb_len > 0:
            state["retrieved_chunks"] = _fuse_subspace_rrf(cur_chunks, fb_retrieved)[:20]
            fused_now = True
            state["fallback_reason"] = (
                f"{fallback_reason} | fused rrf(cur={cur_len},fb={fb_len} -> {len(state['retrieved_chunks'])})"
            )
            log.info(
                "T1 fallback fused: cur=%d (top=%.4f) + fb=%d (top=%.4f) -> %d chunks reason=%s",
                cur_len, cur_top, fb_len, fb_top, len(state["retrieved_chunks"]), fallback_reason,
            )
        elif fb_len > 0:
            # Current was empty — adopt fallback wholesale.
            state["retrieved_chunks"] = fb_retrieved
            state["fallback_reason"] = f"{fallback_reason} | adopted (cur empty)"
            log.info("T1 fallback adopted (cur empty): fallback len=%d top=%.4f", fb_len, fb_top)
        else:
            # Fallback empty — keep original results, reason unchanged.
            state["fallback_reason"] = fallback_reason
            log.info("T1 fallback empty — keeping original %d chunks", cur_len)
        # Merge injected entities if any.
        if fb_injected:
            existing = state.get("injected_entities") or []
            seen = {e.get("list_name") for e in existing}
            merged_inj = list(existing)
            for ent in fb_injected:
                if ent.get("list_name") not in seen:
                    merged_inj.append(ent)
                    seen.add(ent.get("list_name"))
            # Keep merged for audit even if chunks not adopted.
            state["injected_entities"] = merged_inj
            state["supplemental"] = merged_inj
        try:
            from ..tracing import trace_span as _tsp2

            _tsp2(request_id, "retrieve-retry", span_input={"query": search_query, "active_filter": active_filter, "cur_len": cur_len, "cur_top": cur_top, "threshold": thr, "reason": fallback_reason}, span_output={"fb_len": fb_len, "fb_top": fb_top, "fused": fused_now, "fallback_used": True})
        except Exception:
            pass
        return True
    except Exception as e:
        elapsed = round((time.monotonic() - t_retry) * 1000, 1)
        state["fallback_used"] = True
        state["fallback_reason"] = f"{fallback_reason} merge_failed: {e}"
        record_timing(state, "retrieve_retry", elapsed)
        log.warning("T1 fallback merge failed for request %s (keep original): %s", request_id, e)
        try:
            from ..tracing import trace_span as _tsp3

            _tsp3(request_id, "retrieve-retry", span_input={"query": search_query, "active_filter": active_filter, "reason": fallback_reason}, span_output={"error": str(e), "fallback_used": True})
        except Exception:
            pass
        return True


# ---------------------------------------------------------------------------
# T2: Broadened retry when whole-KB fallback still weak
# ---------------------------------------------------------------------------

async def _t2_try_broadened_retry(
    state: dict,
    search_query: str,
    request_id: str,
    top_k: int = 30,
    keyword_boost: float = 1.5,
) -> bool:
    """
    T2: If after T1 the retrieval is still weak (0 chunks or max score <
    evidence threshold) do a second fallback with broadened params:
    whole KB (filter_path=None), top_k=30, keyword_boost=1.5.

    The KB handles synonym beam server-side (KB_SYNONYM_BEAM, default 5);
    the client only needs to pass keyword_boost.  Single broadened retry
    only; guarded by state["broaden_used"].

    Returns True if a broadened attempt was made (even if not adopted).
    Records state["broaden_used"], state["fallback_used"] (also set on
    success), and stage_timing_ms["retrieve_broadened"] plus tracing.
    Reuses _t1_merge_rrf_ce and the same KnowledgebaseClient path (trust_env=False).

    A: when both the previous list and the broadened list are non-empty, the
    two legs are FUSED via weighted RRF (_fuse_subspace_rrf) instead of a
    hard replace; the audit fallback_reason mentions "fused".
    """
    if state.get("broaden_used"):
        return False
    if state.get("faq_verified"):
        return False
    # If still has supplement evidence, don't broaden (supplement counts as evidence)
    # But T1 fallback chunks that are still low DO trigger broaden (they are not supplement).
    if state.get("supplemental") or state.get("injected_entities"):
        # Check if supplemental is non-empty; if it is, evidence exists
        sup = state.get("supplemental") or state.get("injected_entities")
        if sup:
            return False
    cur_chunks = state.get("retrieved_chunks") or []
    cur_top = _compute_top_score(cur_chunks)
    thr = _get_ev_threshold(state.get("applied_subspace"))
    cur_len = len(cur_chunks)
    needs_broaden = (cur_len == 0) or (cur_top < thr)
    if not needs_broaden:
        return False

    t_retry = time.monotonic()
    log.info(
        "T2 broadened retry: still weak after T1 (len=%d top=%.4f < %.3f) — retrying whole KB broadened top_k=%d keyword_boost=%.1f query=%r",
        cur_len, cur_top, thr, top_k, keyword_boost, search_query[:80],
    )
    try:
        from ..clients.knowledgebase import KnowledgebaseClient as _KBClient2

        async with _KBClient2() as fb_client:
            fb_result = await fb_client.retrieve(
                search_query, top_k, request_id, filter_path=None, keyword_boost=keyword_boost
            )
    except Exception as e:
        elapsed = round((time.monotonic() - t_retry) * 1000, 1)
        state["broaden_used"] = True
        # T2 counts as fallback as well for audit (spec: if broadened succeeds, set fallback_used; on failure also mark both so we don't loop)
        state["fallback_used"] = True
        if not state.get("fallback_reason"):
            state["fallback_reason"] = f"broadened_failed: {e}"
        else:
            # append broaden failure to reason without overwriting T1 reason
            state["fallback_reason"] = f"{state['fallback_reason']} | broaden_failed: {e}"
        record_timing(state, "retrieve_broadened", elapsed)
        log.warning("T2 broadened retry failed for request %s (keep original): %s", request_id, e)
        try:
            from ..tracing import trace_span as _tsp
            _tsp(
                request_id,
                "retrieve-broadened",
                span_input={"query": search_query, "cur_len": cur_len, "cur_top": cur_top, "threshold": thr, "top_k": top_k, "keyword_boost": keyword_boost},
                span_output={"error": str(e), "broaden_used": True, "fallback_used": True},
            )
        except Exception:
            pass
        return True

    # Merge broadened result (same RRF+CE union)
    try:
        if isinstance(fb_result, dict):
            fb_rrf = fb_result.get("rrf", [])
            fb_ce = fb_result.get("ce", [])
            fb_injected = fb_result.get("injected_entities", [])
        else:
            fb_rrf = []
            fb_ce = list(fb_result)
            fb_injected = []
        fb_retrieved = _t1_merge_rrf_ce(fb_rrf, fb_ce)
        if state.get("faq_verified") and state.get("direct_faq_answer"):
            fb_retrieved.insert(0, {
                "chunk_id": "__faq_direct__",
                "document_id": "__faq_direct__",
                "title": "FAQ Direct Answer",
                "heading": "FAQ",
                "content": state["direct_faq_answer"],
                "score": 1.0,
                "source": "faq_direct",
                "rank_rrf": 0,
                "rank_ce": 0,
                "hybrid_score": 1.0,
                "rerank_score": None,
            })
        fb_top = _compute_top_score(fb_retrieved)
        fb_len = len(fb_retrieved)
        elapsed = round((time.monotonic() - t_retry) * 1000, 1)
        # Always mark broaden_used and fallback_used (spec: if broadened succeeds, set fallback_used as well)
        state["broaden_used"] = True
        state["fallback_used"] = True
        # Keep original fallback_reason but annotate broadened
        base_reason = state.get("fallback_reason") or f"low_score top={cur_top:.4f} < thr={thr:.3f}"
        # A: fuse the broadened leg with the previous (still weak) results —
        # same weighted RRF as T1 so both legs' recall survives instead of a
        # hard replace. Wholesale replace only when the current list is empty.
        fused_t2 = False
        if cur_len > 0 and fb_len > 0:
            state["retrieved_chunks"] = _fuse_subspace_rrf(cur_chunks, fb_retrieved)[:20]
            fused_t2 = True
            state["fallback_reason"] = (
                f"{base_reason} | broadened top_k={top_k} boost={keyword_boost} -> len={fb_len} top={fb_top:.4f} | fused (cur={cur_len} fb={fb_len} -> {len(state['retrieved_chunks'])})"
            )
            log.info(
                "T2 broadened fused: cur=%d (top=%.4f) + fb=%d (top=%.4f) -> %d chunks",
                cur_len, cur_top, fb_len, fb_top, len(state["retrieved_chunks"]),
            )
        elif fb_len > 0:
            # Current was empty — adopt broadened wholesale.
            state["retrieved_chunks"] = fb_retrieved
            state["fallback_reason"] = f"{base_reason} | broadened top_k={top_k} boost={keyword_boost} -> len={fb_len} top={fb_top:.4f} | adopted (cur empty)"
            log.info("T2 broadened adopted (cur empty): len=%d top=%.4f", fb_len, fb_top)
        else:
            # Broadened empty — keep original results.
            state["fallback_reason"] = f"{base_reason} | broadened top_k={top_k} boost={keyword_boost} -> len=0 (keep original)"
            log.info("T2 broadened empty — keeping original %d chunks", cur_len)
        record_timing(state, "retrieve_broadened", elapsed)
        if fb_injected:
            existing = state.get("injected_entities") or []
            seen = {e.get("list_name") for e in existing}
            merged_inj = list(existing)
            for ent in fb_injected:
                if ent.get("list_name") not in seen:
                    merged_inj.append(ent)
                    seen.add(ent.get("list_name"))
            state["injected_entities"] = merged_inj
            state["supplemental"] = merged_inj
        try:
            from ..tracing import trace_span as _tsp2
            _tsp2(
                request_id,
                "retrieve-broadened",
                span_input={"query": search_query, "cur_len": cur_len, "cur_top": cur_top, "threshold": thr, "top_k": top_k, "keyword_boost": keyword_boost},
                span_output={"fb_len": fb_len, "fb_top": fb_top, "fused": fused_t2, "broaden_used": True, "fallback_used": True},
            )
        except Exception:
            pass
        return True
    except Exception as e:
        elapsed = round((time.monotonic() - t_retry) * 1000, 1)
        state["broaden_used"] = True
        state["fallback_used"] = True
        base = state.get("fallback_reason") or ""
        state["fallback_reason"] = f"{base} | broaden_merge_failed: {e}" if base else f"broaden_merge_failed: {e}"
        record_timing(state, "retrieve_broadened", elapsed)
        log.warning("T2 broadened merge failed for request %s (keep original): %s", request_id, e)
        try:
            from ..tracing import trace_span as _tsp3
            _tsp3(request_id, "retrieve-broadened", span_input={"query": search_query, "top_k": top_k, "keyword_boost": keyword_boost}, span_output={"error": str(e), "broaden_used": True})
        except Exception:
            pass
        return True


async def retrieve(state: RAGState) -> RAGState:
    """
    Rewrite the last user message against history, then call KB search.

    The standalone rewritten query is normalized and used for KB search;
    the original latest user message stays in state["query"] for audit,
    the rewritten form is stored in state["rewritten_query"].
    """
    if state.get("classify") == "greeting":
        state["retrieved_chunks"]=[]; state["rewritten_query"]=state.get("query","")
        return state
    if state.get("classify") == "specification":
        # already set applied_subspace above, no retrieval needed — just ack
        state["retrieved_chunks"]=[]
        # keep the spec text as rewritten_query for audit
        state["rewritten_query"]=state.get("query","")
        # Do not need to ask again; the next question will use the sticky
        # Set a short ack answer that format_response will use? No, let format_response handle spec ack
        state["needs_clarification"]=False
        return state
    t0 = time.monotonic()
    request_id = state["request_id"]
    query = state["query"]
    messages = state.get("messages", [])
    # T1: initialize fallback audit fields (single retry guard)
    if "fallback_used" not in state or state["fallback_used"] is None:
        state["fallback_used"] = False
    if "fallback_reason" not in state or state["fallback_reason"] is None:
        state["fallback_reason"] = None
    # T2: broadened retry audit (second leg after T1)
    if "broaden_used" not in state or state["broaden_used"] is None:
        state["broaden_used"] = False

    # --- Mizan-driven re-retrieval ---
    # If Mizan judged we need to retrieve again or change subspace, apply those params
    mizan_retry = state.get("mizan_retrieve_again")
    mizan_change_subspace = state.get("mizan_change_subspace")
    mizan_retry_params = state.get("mizan_retry_params", {})
    mizan_new_subspace = state.get("mizan_new_subspace")
    
    # Clear Mizan flags so they don't trigger on subsequent retries
    if mizan_retry:
        state["mizan_retrieve_again"] = False
        log.info("Mizan retrieve_again for request %s: params=%s", request_id, mizan_retry_params)
    if mizan_change_subspace:
        state["mizan_change_subspace"] = False
        log.info("Mizan change_subspace for request %s: new_subspace=%s", request_id, mizan_new_subspace)
    
    # If Mizan wants subspace change, override applied_subspace
    if mizan_change_subspace and mizan_new_subspace and mizan_new_subspace != "auto":
        state["applied_subspace"] = mizan_new_subspace
    elif mizan_change_subspace and mizan_new_subspace == "auto":
        # Mizan detected mismatch but didn't specify - will infer from query
        state["applied_subspace"] = None

    faq_answer = find_direct_answer(query)
    if faq_answer:
        state["direct_faq_answer"] = faq_answer
        state["faq_matched"] = True
        state["greeting_only"] = None
        state["rewritten_query"] = query
        # Verify FAQ answer is relevant before using it as context
        try:
            verified = await verify_faq_answer(query, faq_answer)
        except Exception:
            verified = False
        state["faq_verified"] = verified
        if verified:
            log.info("FAQ verified for request %s (query=%r)", request_id, query[:120])
        else:
            log.info("FAQ rejected by verifier for request %s — falling back to retrieval (query=%r)", request_id, query[:120])
    else:
        state["direct_faq_answer"] = None
        state["faq_matched"] = False
        state["faq_verified"] = False

    # Smalltalk short-circuit: greetings/farewell/thanks get a brief reply
    # with NO retrieval and NO history rewrite (a bare سلام must never be
    # rewritten into an older question and answered with cited fabrication).
    kind = detect_greeting_kind(query)
    if kind:
        state["greeting_only"] = kind
        state["rewritten_query"] = query
        state["retrieved_chunks"] = []
        state["injected_entities"] = []
        state["supplemental"] = []
        log.info("Greeting-only (%s) for request %s — skipping retrieval", kind, request_id)
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        record_timing(state, "retrieve", elapsed)
        return state
    state["greeting_only"] = None

    # Stateless memory: resolve coreference ("او", "کجا بوده", ...) against
    # the last 2 exchanges. Falls back to the raw query on any error.
    rewritten = await rewrite_query(messages)
    if not rewritten:
        rewritten = query
    state["rewritten_query"] = rewritten

    try:
        from ..tracing import trace_span
        trace_span(
            request_id,
            "query-rewrite",
            span_input=last_exchanges_text(messages, include_current=True),
            span_output=rewritten,
        )
    except Exception:
        pass

    # Normalize informal Persian: چیه -> چیست for KB search (improves BM25 on short definition queries)
    norm_query = rewritten.replace("چیه", "چیست").replace("چيه", "چیست")
    # Light Persian normalization + boilerplate strip for reworded/conversational
    # queries: unify ZWNJ/space and ي/ك variants so BM25 matches KB wording,
    # and drop politeness filler that dilutes lexical scores.
    norm_query = norm_query.replace("‌", " ").replace("ي", "ی").replace("ك", "ک")
    for filler in (
        "ممکن است توضیح دهید",
        "لطفا دقیق توضیح دهید",
        "لطفاً دقیق توضیح دهید",
        "در عمل",
        "راستی",
        "یه سوال داشتم",
        "یک سوال داشتم",
        "ببخشید",
    ):
        norm_query = norm_query.replace(filler, " ")
    norm_query = " ".join(norm_query.split())
    if not norm_query:
        norm_query = query
    # Keep original query in state for audit, but search with normalized
    search_query = norm_query
    # Stage 1 (instrument): embedding subspace ambiguity, computed ONCE here so
    # every downstream path (sticky, scattered, F5) shares one signal. A scorer
    # failure — missing artifacts, embed service down — must never break
    # retrieval, so it is wrapped and `_amb` stays None. See
    # docs/SUBSPACE_AMBIGUITY.md.
    _amb = None
    try:
        from ..subspaces import get_scorer

        _amb = await get_scorer().ascore(search_query)
        state["ambiguity_score"] = _amb.score
        state["ambiguity_margin"] = _amb.margin
        state["ambiguity_entropy"] = _amb.entropy
        state["ambiguity_ambiguous"] = _amb.ambiguous
        state["ambiguity_source"] = _amb.source
        state["subspace_sims"] = _amb.sims
        state["ambiguity_intra_spread"] = _amb.intra_spread
        state["ambiguity_proto_coverage"] = _amb.proto_coverage
    except Exception as _amb_err:
        log.warning("subspace ambiguity scoring failed: %s", _amb_err)
    settings = get_settings()
    top_k = settings.retrieval_top_k
    # E: dynamic top_k for the PRIMARY retrieval probe — short queries
    # (<= 6 tokens) are likely ambiguous/conversational, so widen the KB
    # candidate pool (top_k=30) to give fusion + cross-encoder rerank more
    # recall headroom. Longer queries keep the default (20). Primary probe
    # only: the T2 broadened retry already uses 30, and the sticky/F5
    # re-retrieves keep the default so per-subspace thresholds stay
    # comparable across paths.
    query_tokens = len(search_query.split())
    primary_top_k = 30 if query_tokens <= 6 else top_k

    # Mizan retry params override (from mizan_retrieval_judge)
    if mizan_retry and mizan_retry_params:
        primary_top_k = mizan_retry_params.get("top_k", primary_top_k)
        # If Mizan says broaden, use whole KB (no filter)
        if mizan_retry_params.get("broaden", False):
            explicit_filter = None
            state["applied_subspace"] = None
            log.info("Mizan broaden=True for request %s - using whole KB", request_id)

    # F7 scope picker: if caller supplied explicit filter_path, honor it
    explicit_filter = (state.get("filter_path") or "").strip() if state.get("filter_path") else None
    if explicit_filter:
        # reverse-map filter_path to subspace label for observability badge
        try:
            _map = settings.subspace_map or {}
            _rev = {v: k for k, v in _map.items()}
            _label = _rev.get(explicit_filter, explicit_filter)
            # shorten common Persian labels for badge
            if "چک" in explicit_filter:
                _label = "cheque"
            elif "حقوقی" in explicit_filter:
                _label = "facilities_legal"
            elif "حقیقی" in explicit_filter:
                _label = "facilities_person"
            state["applied_subspace"] = _label
        except Exception:
            state["applied_subspace"] = explicit_filter
        try:
            async with KnowledgebaseClient() as client:
                # D: per-subspace keyword_boost — cheque Q&A pairs are
                # keyword-dominated and get a stronger BM25 bias (3.5) than
                # facilities (3.0) or general prose (2.5). Only scoped calls
                # pass the boost; whole-KB calls keep the server default.
                result = await client.retrieve(
                    search_query,
                    primary_top_k,
                    request_id,
                    filter_path=explicit_filter,
                    keyword_boost=_subspace_keyword_boost(state.get("applied_subspace") or explicit_filter),
                )
        except Exception as e:
            log.error("KB filtered retrieval failed for request %s (filter=%r): %s — continuing empty", request_id, explicit_filter, e)
            try:
                from ..tracing import trace_span
                trace_span(request_id, "retrieve", span_input={"query": query, "search_query": search_query, "filter_path": explicit_filter}, span_output={"error": str(e), "fallback": "empty"})
            except Exception:
                pass
            state["retrieved_chunks"] = []
            state["injected_entities"] = []
            state["supplemental"] = []
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            record_timing(state, "retrieve", elapsed)
            return state
        # process result same as normal path — jump to shared merge logic below via inline handling
        # We set a flag so later F4/F5 scatter logic is skipped (explicit scope is authoritative)
        _explicit_scope = True
    else:
        _explicit_scope = False
        try:
            async with KnowledgebaseClient() as client:
                result = await client.retrieve(search_query, primary_top_k, request_id)
        except Exception as e:
            log.error("KB retrieval failed for request %s: %s — continuing with empty context", request_id, e)
            try:
                from ..tracing import trace_span
                trace_span(request_id, "retrieve", span_input={"query": query, "search_query": search_query}, span_output={"error": str(e), "fallback": "empty"})
            except Exception:
                pass
            state["retrieved_chunks"] = []
            state["injected_entities"] = []
            state["supplemental"] = []
            elapsed = round((time.monotonic() - t0) * 1000, 1)
            record_timing(state, "retrieve", elapsed)
            return state

    # Client returns {"rrf": [...top10...], "ce": [...top10...], "injected_entities": [...]}; be
    # defensive if an older client shape (plain list) is ever returned.
    if isinstance(result, dict):
        rrf_chunks = result.get("rrf", [])
        ce_chunks = result.get("ce", [])
        injected = result.get("injected_entities", [])
    else:  # pragma: no cover - legacy fallback
        rrf_chunks = []
        ce_chunks = list(result)
        injected = []
    state["injected_entities"] = injected
    state["supplemental"] = injected

    def _item_dict(c, default_source: str) -> dict:
        d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
        get = d.get
        return {
            "chunk_id": get("chunk_id"),
            "document_id": get("document_id"),
            "title": get("title"),
            "heading": get("heading"),
            "content": get("content"),
            "score": get("score"),
            "source": get("source") or default_source,
            "rank_rrf": get("rank_rrf"),
            "rank_ce": get("rank_ce"),
            "hybrid_score": get("hybrid_score"),
            "rerank_score": get("rerank_score"),
            "folder_hierarchy": get("folder_hierarchy"),
        }

    # UNION of top-10 RRF + top-10 CE, deduped by chunk_id. A chunk in both
    # sets becomes a single entry with source="both" + both ranks/scores.
    merged: dict[str, dict] = {}
    order: list[str] = []
    for c in rrf_chunks:
        d = _item_dict(c, "rrf")
        key = d.get("chunk_id") or f"__rrf_{len(order)}"
        if key not in merged:
            merged[key] = d
            order.append(key)
    for c in ce_chunks:
        d = _item_dict(c, "ce")
        key = d.get("chunk_id") or f"__ce_{len(order)}"
        if key in merged:
            existing = merged[key]
            existing["source"] = "both"
            if existing.get("rank_ce") is None:
                existing["rank_ce"] = d.get("rank_ce")
            if existing.get("rerank_score") is None:
                existing["rerank_score"] = d.get("rerank_score")
            if existing.get("hybrid_score") is None:
                existing["hybrid_score"] = d.get("hybrid_score")
            # Prefer the cross-encoder score for the merged entry; fall back
            # to whichever score is present.
            existing["score"] = (
                existing.get("rerank_score")
                if existing.get("rerank_score") is not None
                else d.get("score", existing.get("score"))
            )
            # Fill any missing display fields from the CE copy.
            for k in ("document_id", "title", "heading", "content", "folder_hierarchy"):
                if not existing.get(k) and d.get(k):
                    existing[k] = d[k]
        else:
            merged[key] = d
            order.append(key)

    def _best_rank(d: dict) -> int:
        ranks = [r for r in (d.get("rank_rrf"), d.get("rank_ce")) if isinstance(r, int)]
        return min(ranks) if ranks else 10**9

    both = sorted(
        (merged[k] for k in order if merged[k].get("source") == "both"),
        key=_best_rank,
    )
    ce_only = sorted(
        (merged[k] for k in order if merged[k].get("source") == "ce"),
        key=lambda d: d.get("rank_ce") if isinstance(d.get("rank_ce"), int) else 10**9,
    )
    rrf_only = sorted(
        (merged[k] for k in order if merged[k].get("source") == "rrf"),
        key=lambda d: d.get("rank_rrf") if isinstance(d.get("rank_rrf"), int) else 10**9,
    )
    # Normalize to dict format for state
    retrieved = (both + ce_only + rrf_only)[:20]
    # Prepend FAQ candidate as top-priority context chunk if verified
    if state.get("faq_verified") and state.get("direct_faq_answer"):
        retrieved.insert(0, {
            "chunk_id": "__faq_direct__",
            "document_id": "__faq_direct__",
            "title": "FAQ Direct Answer",
            "heading": "FAQ",
            "content": state["direct_faq_answer"],
            "score": 1.0,
            "source": "faq_direct",
            "rank_rrf": 0,
            "rank_ce": 0,
            "hybrid_score": 1.0,
            "rerank_score": None,
        })
    state["retrieved_chunks"] = retrieved

    log.info(
        "Retrieved %d chunks (rrf=%d, ce=%d, both=%d) for request %s",
        len(state["retrieved_chunks"]), len(rrf_chunks), len(ce_chunks), len(both), request_id,
    )

    # --- T1: Filtered → whole-KB fallback (single retry) ---
    # If the initial probe was filtered (sticky subspace) and evidence is weak
    # (0 chunks or max reranker < 0.15), retry once against the whole KB
    # (1123 chunks, filter_path=None) with same query top_k=20. Record
    # fallback_used/reason and retrieve_retry timing; keep original on failure.
    if _explicit_scope:
        # Explicit filter_path is authoritative and filtered — check T1 condition
        try:
            await _t1_try_whole_kb_fallback(
                state, search_query, request_id, top_k=20, active_filter=explicit_filter
            )
            # If fallback adopted, reflect whole-KB broader context: keep note
            # but do not override applied_subspace if it was explicitly requested
            # for audit transparency (fallback_used still true).
        except Exception as _t1_e:
            log.warning("T1 explicit fallback check failed: %s", _t1_e)

    # --- G5 supplement fallback: if top evidence < threshold, try direct
    # EntityList content search (threshold 0.4 < 0.6) and log outcome (T3 per-subspace) ---
    _ev_threshold = _get_ev_threshold(state.get("applied_subspace") or explicit_filter if _explicit_scope else state.get("applied_subspace"))
    try:
        _top_score = 0.0
        for _c in retrieved:
            for _k in ("rerank_score", "hybrid_score", "score"):
                _v = _c.get(_k)
                if isinstance(_v, (int, float)):
                    _top_score = max(_top_score, float(_v))
        # skip FAQ-verified (already evidence)
        _has_supplement = bool(injected)
        if not state.get("faq_verified") and _top_score < _ev_threshold and not _has_supplement:
            log.info(
                "Low evidence for request %s: top_score=%.4f < %.3f (query=%r) — trying supplementary_search",
                request_id, _top_score, _ev_threshold, search_query[:80],
            )
            try:
                async with KnowledgebaseClient() as _sup_client:
                    _sup = await _sup_client.supplementary_search(search_query, threshold=0.4, request_id=request_id)
                if _sup:
                    log.info("supplementary_search returned %d lists for request %s", len(_sup), request_id)
                    # merge into injected_entities / supplemental
                    existing = state.get("injected_entities") or []
                    seen = {e.get("list_name") for e in existing}
                    merged = list(existing)
                    for ent in _sup:
                        if ent.get("list_name") not in seen:
                            merged.append(ent)
                            seen.add(ent.get("list_name"))
                    state["injected_entities"] = merged
                    state["supplemental"] = merged
                else:
                    log.info("supplementary_search empty for request %s (G5 no supplement)", request_id)
            except Exception as _se:
                log.warning("supplementary_search failed for request %s: %s", request_id, _se)
            try:
                from ..tracing import trace_span as _tsp
                _tsp(
                    request_id,
                    "retrieve-supplement-fallback",
                    span_input={"query": search_query, "top_score": _top_score, "threshold": _ev_threshold},
                    span_output={"supplemental": state.get("supplemental") or state.get("injected_entities") or [], "sup_found": bool(state.get("supplemental"))},
                )
            except Exception:
                pass
    except Exception as _e:
        log.warning("supplement fallback check failed: %s", _e)

    # If still low evidence after supplement, clear for refusal (no hallucinated G5)
    # T1: if whole-KB fallback already provided chunks, keep them even if still low (prefer some evidence over faith 0.0)
    _cur_top_after = _compute_top_score(state.get("retrieved_chunks") or [])
    _cur_thr_after = locals().get("_ev_threshold", 0.15)
    _has_fallback_after = bool(state.get("fallback_used") and len(state.get("retrieved_chunks") or []) > 0)
    # T6: lexical evidence (query/subspace keyword in chunk) bypasses low-score clear
    _has_lex_after = _has_lexical_evidence_retrieve(search_query or query, state.get("retrieved_chunks"), state.get("applied_subspace") or explicit_filter if _explicit_scope else state.get("applied_subspace"))
    if not state.get("faq_verified") and _cur_top_after < _cur_thr_after and not state.get("supplemental") and not state.get("injected_entities") and not _has_fallback_after and not _has_lex_after:
        state["retrieved_chunks"] = []
        log.info("Low evidence still < %.3f after supplement for %s — clear for refusal", _cur_thr_after, request_id)
    elif _has_fallback_after and _cur_top_after < _cur_thr_after:
        log.info("Low evidence still < %.3f but T1 fallback provided %d chunks — keeping for faith (not clearing)", _cur_thr_after, len(state.get("retrieved_chunks") or []))
    elif _has_lex_after and _cur_top_after < _cur_thr_after:
        log.info("T6 lexical evidence keeps %d chunks despite low score %.4f < %.3f", len(state.get("retrieved_chunks") or []), _cur_top_after, _cur_thr_after)

    try:
        from ..tracing import trace_span
        trace_span(
            request_id,
            "retrieve",
            span_input={
                "query": query,
                "rewritten_query": rewritten,
                "search_query": search_query,
                "top_k": top_k,
                "history": last_exchanges_text(messages, include_current=True),
            },
            span_output={
                "rrf_count": len(rrf_chunks),
                "ce_count": len(ce_chunks),
                "both_count": len(both),
                "total": len(state["retrieved_chunks"]),
                "chunks": [
                    {"chunk_id": c.get("chunk_id"), "title": c.get("title"), "score": c.get("score"), "source": c.get("source")}
                    for c in state["retrieved_chunks"][:5]
                ],
            },
        )
    except Exception:
        pass

    # --- Subspace granularity + sticky + clarifying (F4) ---
    # If explicit scope was used, skip F4 inference — the user's picked scope is authoritative
    if _explicit_scope:
        granular = True
        state["needs_clarification"] = False
    else:
        # Sticky subspace persistence: if previous turn set applied_subspace and current
        # message is NOT a new specification, reuse sticky with filtered re-retrieve
        # (skip granularity/clarify). Default None means whole KB (no filter).
        _sticky = state.get("applied_subspace")
        # Current query general override — board member / معرفی signals should bypass sticky cheque
        cur_q2 = state.get("query", "") or ""
        if any(k in cur_q2 for k in ["قاسم", "بور", "هیات مدیره", "هیئت مدیره", "سوالات عمومی", "سوال عمومی", "معرفی شرکت"]):
            state["applied_subspace"] = "general"
            _sticky = "general"
        if _sticky:
            try:
                from .infer_specification import is_specification as _is_spec_sticky

                _last_q_sticky = state.get("query", "")
                try:
                    _is_spec_now = await _is_spec_sticky(_last_q_sticky, request_id)
                except Exception:
                    _is_spec_now = False
                if not _is_spec_now:
                    _filter_path_sticky = get_settings().subspace_map.get(_sticky, "")
                    if _filter_path_sticky:
                        _fresult_sticky = None
                        try:
                            async with KnowledgebaseClient() as _client_sticky:
                                # D: per-subspace keyword_boost on the sticky
                                # scoped re-retrieve (same rationale as the
                                # explicit-scope primary call above).
                                _fresult_sticky = await _client_sticky.retrieve(
                                    search_query,
                                    top_k,
                                    request_id,
                                    filter_path=_filter_path_sticky,
                                    keyword_boost=_subspace_keyword_boost(_sticky or _filter_path_sticky),
                                )
                        except Exception as e:
                            log.warning("Sticky filtered re-retrieve failed: %s", e)
                        if _fresult_sticky is not None:
                            if isinstance(_fresult_sticky, dict):
                                _f_rrf = _fresult_sticky.get("rrf", [])
                                _f_ce = _fresult_sticky.get("ce", [])
                                _f_injected = _fresult_sticky.get("injected_entities", [])
                            else:
                                _f_rrf = []
                                _f_ce = list(_fresult_sticky)
                                _f_injected = []
                            _f_merged: dict[str, dict] = {}
                            _f_order: list[str] = []
                            for c in _f_rrf:
                                d = _item_dict(c, "rrf")
                                key = d.get("chunk_id") or f"__rrf_{len(_f_order)}"
                                if key not in _f_merged:
                                    _f_merged[key] = d
                                    _f_order.append(key)
                            for c in _f_ce:
                                d = _item_dict(c, "ce")
                                key = d.get("chunk_id") or f"__ce_{len(_f_order)}"
                                if key in _f_merged:
                                    existing = _f_merged[key]
                                    existing["source"] = "both"
                                    if existing.get("rank_ce") is None:
                                        existing["rank_ce"] = d.get("rank_ce")
                                    if existing.get("rerank_score") is None:
                                        existing["rerank_score"] = d.get("rerank_score")
                                    if existing.get("hybrid_score") is None:
                                        existing["hybrid_score"] = d.get("hybrid_score")
                                    existing["score"] = (
                                        existing.get("rerank_score")
                                        if existing.get("rerank_score") is not None
                                        else d.get("score", existing.get("score"))
                                    )
                                    for k in ("document_id", "title", "heading", "content", "folder_hierarchy"):
                                        if not existing.get(k) and d.get(k):
                                            existing[k] = d[k]
                                else:
                                    _f_merged[key] = d
                                    _f_order.append(key)

                            def _f_best_rank_sticky(d: dict) -> int:
                                ranks = [r for r in (d.get("rank_rrf"), d.get("rank_ce")) if isinstance(r, int)]
                                return min(ranks) if ranks else 10**9

                            _f_both = sorted(
                                (_f_merged[k] for k in _f_order if _f_merged[k].get("source") == "both"),
                                key=_f_best_rank_sticky,
                            )
                            _f_ce_only = sorted(
                                (_f_merged[k] for k in _f_order if _f_merged[k].get("source") == "ce"),
                                key=lambda d: d.get("rank_ce") if isinstance(d.get("rank_ce"), int) else 10**9,
                            )
                            _f_rrf_only = sorted(
                                (_f_merged[k] for k in _f_order if _f_merged[k].get("source") == "rrf"),
                                key=lambda d: d.get("rank_rrf") if isinstance(d.get("rank_rrf"), int) else 10**9,
                            )
                            _filtered_retrieved = (_f_both + _f_ce_only + _f_rrf_only)[:20]
                            if state.get("faq_verified") and state.get("direct_faq_answer"):
                                _filtered_retrieved.insert(
                                    0,
                                    {
                                        "chunk_id": "__faq_direct__",
                                        "document_id": "__faq_direct__",
                                        "title": "FAQ Direct Answer",
                                        "heading": "FAQ",
                                        "content": state["direct_faq_answer"],
                                        "score": 1.0,
                                        "source": "faq_direct",
                                        "rank_rrf": 0,
                                        "rank_ce": 0,
                                        "hybrid_score": 1.0,
                                        "rerank_score": None,
                                    },
                                )
                            state["retrieved_chunks"] = _filtered_retrieved
                            if _f_injected:
                                existing_injected = state.get("injected_entities") or []
                                seen = {e.get("list_name") for e in existing_injected}
                                merged_injected = list(existing_injected)
                                for ent in _f_injected:
                                    if ent.get("list_name") not in seen:
                                        merged_injected.append(ent)
                                        seen.add(ent.get("list_name"))
                                state["injected_entities"] = merged_injected
                                state["supplemental"] = merged_injected
                            state["needs_clarification"] = False
                            # --- T1: filtered → whole-KB fallback (single retry, before LLM judge broadening) ---
                            # T4: For generic follow-ups that inherit sticky (no explicit re-spec), do NOT fallback to whole KB - keep sticky
                            # Only fallback when there's explicit hint of different subspace; otherwise generic keeps sticky even if weak
                            try:
                                if not state.get("fallback_used"):
                                    _t1_chk_len = len(state.get("retrieved_chunks") or [])
                                    _t1_chk_top = _compute_top_score(state.get("retrieved_chunks"))
                                    _t1_chk_thr = _get_ev_threshold(_filter_path_sticky or _sticky)
                                    _t1_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
                                    _t1_needs = (_t1_chk_len == 0 or _t1_chk_top < _t1_chk_thr) and not state.get("faq_verified") and not _t1_has_sup
                                    # T4 gate: check if current query explicitly mentions a different subspace before allowing fallback
                                    if _t1_needs:
                                        _cur_q_t1 = state.get("query", "") or ""
                                        # Generic follow-up has no explicit different subspace keyword -> preserve sticky, skip fallback
                                        _has_different_kw = False
                                        if _sticky == "cheque" and ("تسهیلات" in _cur_q_t1 and ("حقیقی" in _cur_q_t1 or "حقوقی" in _cur_q_t1)):
                                            _has_different_kw = True
                                        elif _sticky == "facilities_person" and "چک" in _cur_q_t1 and len(_cur_q_t1.split()) <= 6:
                                            # short "چک" after facilities_person is explicit re-spec via classify spec path, not via sticky weak fallback
                                            # For longer generic like "تاثیر چک..." keep sticky, so only short counts
                                            _has_different_kw = True
                                        elif _sticky == "facilities_legal" and "چک" in _cur_q_t1 and len(_cur_q_t1.split()) <= 6:
                                            _has_different_kw = True
                                        # Only fallback if explicit different subspace hint present; otherwise keep sticky
                                        if _has_different_kw:
                                            await _t1_try_whole_kb_fallback(state, search_query, request_id, top_k=20, active_filter=_filter_path_sticky)
                                            # If fallback made evidence sufficient, skip further broadening
                                            _post_top = _compute_top_score(state.get("retrieved_chunks") or [])
                                            _post_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
                                            if _post_top >= _t1_chk_thr or _post_has_sup:
                                                try:
                                                    state["retrieve_attempts"] = int(state.get("retrieve_attempts") or 0) + 1
                                                except Exception:
                                                    state["retrieve_attempts"] = 1
                                                _t1_elapsed = round((time.monotonic() - t0) * 1000, 1)
                                                if "stage_timing_ms" not in state or state["stage_timing_ms"] is None:
                                                    state["stage_timing_ms"] = {}
                                                state["stage_timing_ms"]["retrieve"] = _t1_elapsed
                                                try:
                                                    from ..tracing import trace_span as _tsp_t1
                                                    _tsp_t1(request_id, "retrieve", span_input={"query": query, "search_query": search_query, "filter_path": _filter_path_sticky, "sticky": _sticky, "t1_fallback": True}, span_output={"total": len(state.get("retrieved_chunks") or []), "sticky_filtered": True, "fallback_used": True})
                                                except Exception:
                                                    pass
                                                log.info("T1 sticky fallback sufficient — skipping LLM broadening, returning %d chunks", len(state.get("retrieved_chunks") or []))
                                                return state
                                        else:
                                            log.info("T4 T1 fallback skipped for generic follow-up %s (weak top=%.4f < %.3f but no explicit different keyword) - keeping sticky", _sticky, _t1_chk_top, _t1_chk_thr)
                            except Exception as _t1e:
                                log.warning("T1 sticky fallback check failed: %s", _t1e)
                            # --- T4 Dynamic broadening: filtered empty -> LLM judge then preserve sticky for generic (T4) ---
                            _broad_thr = _get_ev_threshold(_filter_path_sticky or _sticky)
                            # If T1 already retried whole KB, don't broaden again (single retry budget)
                            if state.get("fallback_used"):
                                _broad_empty = False
                            else:
                                _broad_empty = len(_filtered_retrieved) == 0 or _compute_top_score(_filtered_retrieved) < _broad_thr
                            if _broad_empty:
                                log.info("Filtered retrieval empty for subspace %s, broadening to whole KB", _sticky)
                                try:
                                    from .infer_specification import infer_subspace_llm
                                    from ..rewrite import last_exchanges_text
                                    hist = last_exchanges_text(state.get("messages", []) or [], include_current=False)
                                    llm_choice = await infer_subspace_llm(state.get("query", ""), hist, state["request_id"])
                                    if llm_choice and llm_choice in get_settings().subspace_map and llm_choice != _sticky:
                                        new_filter = get_settings().subspace_map[llm_choice]
                                        _broad_result = None
                                        try:
                                            async with KnowledgebaseClient() as _client2:
                                                _broad_result = await _client2.retrieve(search_query, top_k, request_id, filter_path=new_filter)
                                        except Exception as _e2:
                                            log.warning("LLM broadening re-retrieve failed: %s", _e2)
                                        if _broad_result is not None:
                                            if isinstance(_broad_result, dict):
                                                _b_rrf = _broad_result.get("rrf", [])
                                                _b_ce = _broad_result.get("ce", [])
                                                _b_injected = _broad_result.get("injected_entities", [])
                                            else:
                                                _b_rrf = []
                                                _b_ce = list(_broad_result)
                                                _b_injected = []
                                            _b_merged: dict[str, dict] = {}
                                            _b_order: list[str] = []
                                            for c in _b_rrf:
                                                d = _item_dict(c, "rrf")
                                                key = d.get("chunk_id") or f"__rrf_{len(_b_order)}"
                                                if key not in _b_merged:
                                                    _b_merged[key] = d
                                                    _b_order.append(key)
                                            for c in _b_ce:
                                                d = _item_dict(c, "ce")
                                                key = d.get("chunk_id") or f"__ce_{len(_b_order)}"
                                                if key in _b_merged:
                                                    existing = _b_merged[key]
                                                    existing["source"] = "both"
                                                    if existing.get("rank_ce") is None:
                                                        existing["rank_ce"] = d.get("rank_ce")
                                                    if existing.get("rerank_score") is None:
                                                        existing["rerank_score"] = d.get("rerank_score")
                                                    if existing.get("hybrid_score") is None:
                                                        existing["hybrid_score"] = d.get("hybrid_score")
                                                    existing["score"] = (
                                                        existing.get("rerank_score")
                                                        if existing.get("rerank_score") is not None
                                                        else d.get("score", existing.get("score"))
                                                    )
                                                    for k in ("document_id", "title", "heading", "content", "folder_hierarchy"):
                                                        if not existing.get(k) and d.get(k):
                                                            existing[k] = d[k]
                                                else:
                                                    _b_merged[key] = d
                                                    _b_order.append(key)
                                            def _b_best_rank(d: dict) -> int:
                                                ranks = [r for r in (d.get("rank_rrf"), d.get("rank_ce")) if isinstance(r, int)]
                                                return min(ranks) if ranks else 10**9
                                            _b_both = sorted((_b_merged[k] for k in _b_order if _b_merged[k].get("source") == "both"), key=_b_best_rank)
                                            _b_ce_only = sorted((_b_merged[k] for k in _b_order if _b_merged[k].get("source") == "ce"), key=lambda d: d.get("rank_ce") if isinstance(d.get("rank_ce"), int) else 10**9)
                                            _b_rrf_only = sorted((_b_merged[k] for k in _b_order if _b_merged[k].get("source") == "rrf"), key=lambda d: d.get("rank_rrf") if isinstance(d.get("rank_rrf"), int) else 10**9)
                                            new_filtered = (_b_both + _b_ce_only + _b_rrf_only)[:20]
                                            if state.get("faq_verified") and state.get("direct_faq_answer"):
                                                new_filtered.insert(0, {"chunk_id": "__faq_direct__", "document_id": "__faq_direct__", "title": "FAQ Direct Answer", "heading": "FAQ", "content": state["direct_faq_answer"], "score": 1.0, "source": "faq_direct", "rank_rrf": 0, "rank_ce": 0, "hybrid_score": 1.0, "rerank_score": None})
                                            if _b_injected:
                                                existing_injected = state.get("injected_entities") or []
                                                seen = {e.get("list_name") for e in existing_injected}
                                                merged_injected = list(existing_injected)
                                                for ent in _b_injected:
                                                    if ent.get("list_name") not in seen:
                                                        merged_injected.append(ent)
                                                        seen.add(ent.get("list_name"))
                                                state["injected_entities"] = merged_injected
                                                state["supplemental"] = merged_injected
                                            state["applied_subspace"] = llm_choice
                                            state["retrieved_chunks"] = new_filtered
                                            log.info("Broadened from %s to %s: %d chunks", _sticky, llm_choice, len(new_filtered))
                                            return state
                                except Exception as e:
                                    log.warning("Broadening failed: %s", e)
                                # T4: Generic follow-up should keep sticky, not drop to whole KB
                                # If llm_choice is None/unclear or same as sticky, it's a generic follow-up -> preserve sticky filtered results
                                # Only whole-KB fallback would drop sticky to None ( ∅ ), which breaks persistence for "تاثیر چک..." and "آیا میزان درآمد..."
                                _llm_choice_local = locals().get("llm_choice", None)
                                _is_generic_preserve = (_llm_choice_local is None or _llm_choice_local == _sticky or _llm_choice_local not in get_settings().subspace_map)
                                if _is_generic_preserve:
                                    log.info("T4 generic preserve sticky %s (llm_choice=%s) weak filtered len=%d top=%.4f - keeping sticky, not broadening to whole KB", _sticky, _llm_choice_local, len(_filtered_retrieved), _compute_top_score(_filtered_retrieved))
                                    # Preserve sticky filtered results (even if weak) - ensure applied_subspace stays, record timing and return
                                    try:
                                        state["retrieve_attempts"] = int(state.get("retrieve_attempts") or 0) + 1
                                    except Exception:
                                        state["retrieve_attempts"] = 1
                                    _gen_elapsed = round((time.monotonic() - t0) * 1000, 1)
                                    if "stage_timing_ms" not in state or state["stage_timing_ms"] is None:
                                        state["stage_timing_ms"] = {}
                                    state["stage_timing_ms"]["retrieve"] = _gen_elapsed
                                    try:
                                        from ..tracing import trace_span as _tsp_gen
                                        _tsp_gen(request_id, "retrieve", span_input={"query": query, "search_query": search_query, "filter_path": _filter_path_sticky, "sticky": _sticky, "llm_choice": _llm_choice_local, "t4_preserve": True}, span_output={"total": len(state.get("retrieved_chunks") or []), "sticky_filtered": True, "t4_generic_keep": True})
                                    except Exception:
                                        pass
                                    return state
                                # Fall back to whole KB (no filter) - only for explicit cases where llm_choice different but switch failed, or broadening logic requires whole KB
                                try:
                                    _whole_result = None
                                    async with KnowledgebaseClient() as _client3:
                                        _whole_result = await _client3.retrieve(search_query, top_k, request_id, filter_path=None)
                                    if _whole_result is not None:
                                        if isinstance(_whole_result, dict):
                                            _w_rrf = _whole_result.get("rrf", [])
                                            _w_ce = _whole_result.get("ce", [])
                                            _w_injected = _whole_result.get("injected_entities", [])
                                        else:
                                            _w_rrf = []
                                            _w_ce = list(_whole_result)
                                            _w_injected = []
                                        _w_merged: dict[str, dict] = {}
                                        _w_order: list[str] = []
                                        for c in _w_rrf:
                                            d = _item_dict(c, "rrf")
                                            key = d.get("chunk_id") or f"__rrf_{len(_w_order)}"
                                            if key not in _w_merged:
                                                _w_merged[key] = d
                                                _w_order.append(key)
                                        for c in _w_ce:
                                            d = _item_dict(c, "ce")
                                            key = d.get("chunk_id") or f"__ce_{len(_w_order)}"
                                            if key in _w_merged:
                                                existing = _w_merged[key]
                                                existing["source"] = "both"
                                                if existing.get("rank_ce") is None:
                                                    existing["rank_ce"] = d.get("rank_ce")
                                                if existing.get("rerank_score") is None:
                                                    existing["rerank_score"] = d.get("rerank_score")
                                                if existing.get("hybrid_score") is None:
                                                    existing["hybrid_score"] = d.get("hybrid_score")
                                                existing["score"] = (
                                                    existing.get("rerank_score")
                                                    if existing.get("rerank_score") is not None
                                                    else d.get("score", existing.get("score"))
                                                )
                                                for k in ("document_id", "title", "heading", "content", "folder_hierarchy"):
                                                    if not existing.get(k) and d.get(k):
                                                        existing[k] = d[k]
                                            else:
                                                _w_merged[key] = d
                                                _w_order.append(key)
                                        def _w_best_rank(d: dict) -> int:
                                            ranks = [r for r in (d.get("rank_rrf"), d.get("rank_ce")) if isinstance(r, int)]
                                            return min(ranks) if ranks else 10**9
                                        _w_both = sorted((_w_merged[k] for k in _w_order if _w_merged[k].get("source") == "both"), key=_w_best_rank)
                                        _w_ce_only = sorted((_w_merged[k] for k in _w_order if _w_merged[k].get("source") == "ce"), key=lambda d: d.get("rank_ce") if isinstance(d.get("rank_ce"), int) else 10**9)
                                        _w_rrf_only = sorted((_w_merged[k] for k in _w_order if _w_merged[k].get("source") == "rrf"), key=lambda d: d.get("rank_rrf") if isinstance(d.get("rank_rrf"), int) else 10**9)
                                        whole_retrieved = (_w_both + _w_ce_only + _w_rrf_only)[:20]
                                        if state.get("faq_verified") and state.get("direct_faq_answer"):
                                            whole_retrieved.insert(0, {"chunk_id": "__faq_direct__", "document_id": "__faq_direct__", "title": "FAQ Direct Answer", "heading": "FAQ", "content": state["direct_faq_answer"], "score": 1.0, "source": "faq_direct", "rank_rrf": 0, "rank_ce": 0, "hybrid_score": 1.0, "rerank_score": None})
                                        if _w_injected:
                                            existing_injected = state.get("injected_entities") or []
                                            seen = {e.get("list_name") for e in existing_injected}
                                            merged_injected = list(existing_injected)
                                            for ent in _w_injected:
                                                if ent.get("list_name") not in seen:
                                                    merged_injected.append(ent)
                                                    seen.add(ent.get("list_name"))
                                            state["injected_entities"] = merged_injected
                                            state["supplemental"] = merged_injected
                                        state["applied_subspace"] = None
                                        state["retrieved_chunks"] = whole_retrieved
                                        # T1 audit: this whole-KB broaden counts as fallback if not already marked
                                        if not state.get("fallback_used"):
                                            state["fallback_used"] = True
                                            _wb_top = _compute_top_score(whole_retrieved)
                                            _wb_orig_top = _compute_top_score(_filtered_retrieved)
                                            state["fallback_reason"] = f"broadened whole-KB fallback top={_wb_top:.4f} orig={_wb_orig_top:.4f} sub={_sticky}"
                                            # Approximate retry timing from t0 (or record 0 if not measured)
                                            try:
                                                _wb_elapsed = round((time.monotonic() - t0) * 1000, 1)
                                                # Only set retrieve_retry if not already set by T1 helper
                                                if not state.get("stage_timing_ms", {}).get("retrieve_retry"):
                                                    record_timing(state, "retrieve_retry", _wb_elapsed)
                                            except Exception:
                                                pass
                                        log.info("Broadened to whole KB: %d chunks", len(whole_retrieved))
                                        return state
                                except Exception as _e3:
                                    log.warning("Whole KB fallback failed: %s", _e3)
                                # If broadened still empty, fall through to DLQ/clear logic below
                            # --- Dynamic broadening: one re-retrieval then not-enough-context + DLQ ---
                            # Increment attempts (sticky counts as the single allowed re-retrieve)
                            try:
                                state["retrieve_attempts"] = int(state.get("retrieve_attempts") or 0) + 1
                            except Exception:
                                state["retrieve_attempts"] = 1
                            _sticky_top = _compute_top_score(state["retrieved_chunks"])
                            _sticky_thr = _get_ev_threshold(_filter_path_sticky or _sticky)
                            _sticky_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
                            # T1: if whole-KB fallback already gave us chunks, keep them even if still low (better than empty faith 0.0)
                            _has_fallback_chunks = bool(state.get("fallback_used") and len(state.get("retrieved_chunks") or []) > 0)
                            # T6 lexical bypass
                            _sticky_has_lex = _has_lexical_evidence_retrieve(search_query or query, state.get("retrieved_chunks"), _filter_path_sticky or _sticky)
                            if _sticky_has_lex:
                                log.info("T6 sticky lexical evidence keeps %d chunks despite low score %.4f < %.3f", len(state.get("retrieved_chunks") or []), _sticky_top, _sticky_thr)
                            if not state.get("faq_verified") and _sticky_top < _sticky_thr and not _sticky_has_sup and not _has_fallback_chunks and not _sticky_has_lex:
                                # Do NOT retry again — DLQ + clear for refusal (single retry budget exhausted)
                                state["retrieved_chunks"] = []
                                state["dlq_saved"] = True
                                try:
                                    from ..tracing import trace_span as _tsp2
                                    _tsp2(request_id, "retrieve-dlq", span_input={"query": query, "search_query": search_query, "top_score": _sticky_top, "threshold": _sticky_thr, "stage": "sticky_filtered", "retrieve_attempts": state.get("retrieve_attempts")}, span_output={"dlq": True, "query": query, "cleared": True})
                                except Exception:
                                    pass
                                try:
                                    import json as _json2, time as _time2
                                    rec2 = {"ts": _time2.time(), "request_id": request_id, "query": query, "search_query": search_query, "top_score": _sticky_top, "threshold": _sticky_thr, "stage": "sticky_filtered", "dlq": True, "retrieve_attempts": state.get("retrieve_attempts")}
                                    with open(_DLQ_PATH, "a", encoding="utf-8") as fh2:
                                        fh2.write(_json2.dumps(rec2, ensure_ascii=False) + "\n")
                                except Exception:
                                    pass
                                log.info("Sticky re-retrieve still low (%.4f < %.3f) — cleared for refusal + DLQ", _sticky_top, _sticky_thr)
                            log.info(
                                "Sticky filtered re-retrieve for subspace %s (%s): %d chunks",
                                _sticky,
                                _filter_path_sticky,
                                len(_filtered_retrieved) if not state.get("dlq_saved") else 0,
                            )
                            elapsed = round((time.monotonic() - t0) * 1000, 1)
                            if "stage_timing_ms" not in state or state["stage_timing_ms"] is None:
                                state["stage_timing_ms"] = {}
                            state["stage_timing_ms"]["retrieve"] = elapsed
                            try:
                                from ..tracing import trace_span

                                trace_span(
                                    request_id,
                                    "retrieve",
                                    span_input={
                                        "query": query,
                                        "rewritten_query": rewritten,
                                        "search_query": search_query,
                                        "top_k": top_k,
                                        "filter_path": _filter_path_sticky,
                                        "sticky": _sticky,
                                    },
                                    span_output={
                                        "total": len(state["retrieved_chunks"]),
                                        "sticky_filtered": True,
                                        "dlq_saved": bool(state.get("dlq_saved")),
                                    },
                                )
                            except Exception:
                                pass
                            return state
            except Exception as e:
                log.warning("Sticky check failed: %s", e)
        granular = True
        # T5 scattered gate: short generic vague without subspace keywords => force clarify before threshold logic
        try:
            from .infer_specification import is_scattered_ambiguous as _is_scattered_ret

            _q_sc = state.get("query", "") or ""
            _has_sticky_sc = bool(state.get("applied_subspace"))
            if not _has_sticky_sc and _is_scattered_ret(_q_sc):
                state["needs_clarification"] = True
                state["available_subspaces"] = list(SUBSPACES)
                state["clarifying_question"] = None
                # keep applied_subspace as None to signal no clear subspace
                state["applied_subspace"] = None
                log.info("T5 scattered ambiguous detected in retrieve -> clarify q=%r", _q_sc[:80])
                elapsed_sc = round((time.monotonic() - t0) * 1000, 1)
                record_timing(state, "retrieve", elapsed_sc)
                try:
                    from ..tracing import trace_span as _tsp_sc

                    _tsp_sc(state["request_id"], "retrieve", span_input={"query": state.get("query"), "search_query": search_query, "scattered": True}, span_output={"needs_clarification": True, "available_subspaces": state["available_subspaces"]})
                except Exception:
                    pass
                return state
        except Exception as _sc_e:
            log.warning("T5 scattered check failed: %s", _sc_e)
        try:
            from collections import Counter

            subspaces = [_extract_subspace(c.get("folder_hierarchy", [])) for c in state.get("retrieved_chunks", [])]
            cnt = Counter(subspaces)
            if cnt:
                most, n = cnt.most_common(1)[0]
                total = len(subspaces)
                try:
                    _s = get_settings()
                    _thresh = float(_s.subspace_threshold)
                    _general_exempt = bool(_s.subspace_general_exempt)
                except Exception:
                    _thresh = 0.8
                    _general_exempt = True
                last_q = state.get("query", "")
                is_general = most == "general"
                # General exempt only when query itself looks general (avoid suppressing clarify for generic credit queries like "وضعیت اعتباری چطور است")
                _is_general_query = any(k in last_q for k in ["عمومی","عام","معرفی","مدیر","هیات مدیره","هیئت مدیره","کاربرد","قاسم","بور","سوالات عمومی","سوال عمومی"])
                # T8: if query already has explicit subspace keyword, treat as granular even if retrieval distribution is mixed (fixes over-clarify for granular singles like facilities_person with 0.03 scores)
                _has_explicit = any(kw in last_q for kw in ["چک","تسهیلات","وام","قسط","حقیقی","حقوقی","شرکت","سهامدار","معرفی","مدیر","هیات مدیره","هیئت مدیره","عمومی","عام","درآمد","گردش حساب","گردش"])
                if _has_explicit and not is_general:
                    granular = True
                else:
                    granular = (is_general and _general_exempt and _is_general_query) or (n / total >= _thresh)
                # G5 fake grade must never clarify — force granular to go to guard instead
                if "G5" in last_q or "G 5" in last_q:
                    granular = True
                state["needs_clarification"] = False
                from .infer_specification import is_specification
                try:
                    is_spec = await is_specification(last_q, state["request_id"])
                except Exception:
                    is_spec = False
                if is_spec:
                    low = last_q.strip().lower()
                    # "هیچ کدوم" + manager/company → general (user explicitly rejected credit options)
                    if ("هیچ کدوم" in last_q or "هیچکدوم" in last_q) and any(kw in last_q for kw in ["مدیر","هیات مدیره","هیئت مدیره","معرفی","عمومی","شرکت","عام"]):
                        state["applied_subspace"] = "general"
                    elif "چک" in low:
                        state["applied_subspace"] = "cheque"
                    elif any(kw in last_q for kw in ["مدیر","هیات مدیره","هیئت مدیره","معرفی","عمومی","عام","کاربرد"]) or "general" in low:
                        state["applied_subspace"] = "general"
                    elif "حقوقی" in low or ("شرکت" in last_q and ("تسهیلات" in last_q or "وام" in last_q)):
                        state["applied_subspace"] = "facilities_legal"
                    elif "شرکت" in last_q and "تسهیلات" not in last_q and "چک" not in last_q:
                        # "شرکت" alone without تسهیلات/چک → عمومی (معرفی شرکت)
                        state["applied_subspace"] = "general"
                    else:
                        state["applied_subspace"] = "facilities_person"
                    state["needs_clarification"] = False
                elif not granular:
                    sticky = state.get("applied_subspace")
                    if sticky:
                        # Use sticky — will trigger re-retrieve in F5
                        pass
                    else:
                        # Scope-less scattered question -> ask LLM judge, not keyword table
                        try:
                            from .infer_specification import infer_subspace_llm
                            from ..rewrite import last_exchanges_text
                            hist = last_exchanges_text(state.get("messages",[]), include_current=False)
                            llm_choice = await infer_subspace_llm(state.get("query",""), hist, state["request_id"])
                        except:
                            llm_choice = None
                        # Guard: generic ambiguous queries like "وضعیت اعتباری چطور است" have no subspace keyword — treat LLM leaf as unclear to force clarification
                        _q_for_llm = state.get("query", "") or ""
                        _has_hint = any(kw in _q_for_llm for kw in ["چک","صیاد","پیچک","برگشت","تسهیلات","وام","قسط","حقوقی","شرکت","سهامدار","معرفی","مدیر","هیات مدیره","هیئت مدیره","عمومی","عام","کاربرد"])
                        if llm_choice and not _has_hint:
                            llm_choice = None
                        # Ambiguous co-occurrence: چک + تاثیر/اعتبارسنجی/تسهیلات/امتیاز → unclear, must clarify
                        if llm_choice:
                            _has_cheque = any(kw in _q_for_llm for kw in ["چک", "صیاد", "پیچک", "برگشت"])
                            _has_credit_effect = any(kw in _q_for_llm for kw in ["تاثیر", "تأثیر", "اعتبارسنجی", "تسهیلات", "امتیاز"])
                            if _has_cheque and _has_credit_effect:
                                llm_choice = None
                        if llm_choice and llm_choice in SUBSPACES:
                            filter_path = get_settings().subspace_map.get(llm_choice, "")
                            if filter_path:
                                _fresult_llm = None
                                try:
                                    async with KnowledgebaseClient() as _client_llm:
                                        _fresult_llm = await _client_llm.retrieve(
                                    search_query, top_k, request_id,
                                    filter_path=filter_path,
                                    keyword_boost=_subspace_keyword_boost(llm_choice or filter_path),
                                )
                                except Exception as e:
                                    log.warning("LLM-judge filtered re-retrieve failed: %s", e)
                                if _fresult_llm is not None:
                                    if isinstance(_fresult_llm, dict):
                                        _f_rrf = _fresult_llm.get("rrf", [])
                                        _f_ce = _fresult_llm.get("ce", [])
                                        _f_injected = _fresult_llm.get("injected_entities", [])
                                    else:
                                        _f_rrf = []
                                        _f_ce = list(_fresult_llm)
                                        _f_injected = []
                                    _f_merged: dict[str, dict] = {}
                                    _f_order: list[str] = []
                                    for c in _f_rrf:
                                        d = _item_dict(c, "rrf")
                                        key = d.get("chunk_id") or f"__rrf_{len(_f_order)}"
                                        if key not in _f_merged:
                                            _f_merged[key] = d
                                            _f_order.append(key)
                                    for c in _f_ce:
                                        d = _item_dict(c, "ce")
                                        key = d.get("chunk_id") or f"__ce_{len(_f_order)}"
                                        if key in _f_merged:
                                            existing = _f_merged[key]
                                            existing["source"] = "both"
                                            if existing.get("rank_ce") is None:
                                                existing["rank_ce"] = d.get("rank_ce")
                                            if existing.get("rerank_score") is None:
                                                existing["rerank_score"] = d.get("rerank_score")
                                            if existing.get("hybrid_score") is None:
                                                existing["hybrid_score"] = d.get("hybrid_score")
                                            existing["score"] = (
                                                existing.get("rerank_score")
                                                if existing.get("rerank_score") is not None
                                                else d.get("score", existing.get("score"))
                                            )
                                            for k in ("document_id", "title", "heading", "content", "folder_hierarchy"):
                                                if not existing.get(k) and d.get(k):
                                                    existing[k] = d[k]
                                        else:
                                            _f_merged[key] = d
                                            _f_order.append(key)
                                    def _f_best_rank_llm(d: dict) -> int:
                                        ranks = [r for r in (d.get("rank_rrf"), d.get("rank_ce")) if isinstance(r, int)]
                                        return min(ranks) if ranks else 10**9
                                    _f_both = sorted(
                                        (_f_merged[k] for k in _f_order if _f_merged[k].get("source") == "both"),
                                        key=_f_best_rank_llm,
                                    )
                                    _f_ce_only = sorted(
                                        (_f_merged[k] for k in _f_order if _f_merged[k].get("source") == "ce"),
                                        key=lambda d: d.get("rank_ce") if isinstance(d.get("rank_ce"), int) else 10**9,
                                    )
                                    _f_rrf_only = sorted(
                                        (_f_merged[k] for k in _f_order if _f_merged[k].get("source") == "rrf"),
                                        key=lambda d: d.get("rank_rrf") if isinstance(d.get("rank_rrf"), int) else 10**9,
                                    )
                                    _filtered_retrieved = (_f_both + _f_ce_only + _f_rrf_only)[:20]
                                    if state.get("faq_verified") and state.get("direct_faq_answer"):
                                        _filtered_retrieved.insert(0, {
                                            "chunk_id": "__faq_direct__",
                                            "document_id": "__faq_direct__",
                                            "title": "FAQ Direct Answer",
                                            "heading": "FAQ",
                                            "content": state["direct_faq_answer"],
                                            "score": 1.0,
                                            "source": "faq_direct",
                                            "rank_rrf": 0,
                                            "rank_ce": 0,
                                            "hybrid_score": 1.0,
                                            "rerank_score": None,
                                        })
                                    state["retrieved_chunks"] = _filtered_retrieved
                                    if _f_injected:
                                        existing_injected = state.get("injected_entities") or []
                                        seen = {e.get("list_name") for e in existing_injected}
                                        merged_injected = list(existing_injected)
                                        for ent in _f_injected:
                                            if ent.get("list_name") not in seen:
                                                merged_injected.append(ent)
                                                seen.add(ent.get("list_name"))
                                        state["injected_entities"] = merged_injected
                                        state["supplemental"] = merged_injected
                                    state["applied_subspace"] = llm_choice
                                    state["needs_clarification"] = False
                                    # --- Dynamic broadening: one re-retrieval then not-enough-context + DLQ ---
                                    try:
                                        state["retrieve_attempts"] = int(state.get("retrieve_attempts") or 0) + 1
                                    except Exception:
                                        state["retrieve_attempts"] = 1
                                    _llm_top = _compute_top_score(state["retrieved_chunks"])
                                    _llm_thr = _get_ev_threshold(llm_choice)
                                    _llm_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
                                    _llm_has_lex = _has_lexical_evidence_retrieve(search_query or query, state.get("retrieved_chunks"), llm_choice)
                                    if _llm_has_lex:
                                        log.info("T6 llm-judge lexical evidence keeps %d chunks despite low score %.4f < %.3f", len(state.get("retrieved_chunks") or []), _llm_top, _llm_thr)
                                    if not state.get("faq_verified") and _llm_top < _llm_thr and not _llm_has_sup and not _llm_has_lex:
                                        state["retrieved_chunks"] = []
                                        state["dlq_saved"] = True
                                        try:
                                            from ..tracing import trace_span as _tsp3
                                            _tsp3(request_id, "retrieve-dlq", span_input={"query": query, "search_query": search_query, "top_score": _llm_top, "threshold": _llm_thr, "stage": "llm_judge_filtered", "retrieve_attempts": state.get("retrieve_attempts")}, span_output={"dlq": True, "query": query, "cleared": True})
                                        except Exception:
                                            pass
                                        try:
                                            import json as _json3, time as _time3
                                            rec3 = {"ts": _time3.time(), "request_id": request_id, "query": query, "search_query": search_query, "top_score": _llm_top, "threshold": _llm_thr, "stage": "llm_judge_filtered", "dlq": True, "retrieve_attempts": state.get("retrieve_attempts")}
                                            with open(_DLQ_PATH, "a", encoding="utf-8") as fh3:
                                                fh3.write(_json3.dumps(rec3, ensure_ascii=False) + "\n")
                                        except Exception:
                                            pass
                                        log.info("LLM-judge re-retrieve still low (%.4f < %.3f) — cleared for refusal + DLQ", _llm_top, _llm_thr)
                                    log.info("LLM-judge filtered re-retrieve for subspace %s (%s): %d chunks", llm_choice, filter_path, len(_filtered_retrieved) if not state.get("dlq_saved") else 0)
                                    elapsed = round((time.monotonic() - t0) * 1000, 1)
                                    if "stage_timing_ms" not in state or state["stage_timing_ms"] is None:
                                        state["stage_timing_ms"] = {}
                                    state["stage_timing_ms"]["retrieve"] = elapsed
                                    try:
                                        from ..tracing import trace_span
                                        trace_span(request_id, "retrieve", span_input={"query": query, "rewritten_query": rewritten, "search_query": search_query, "top_k": top_k, "filter_path": filter_path, "llm_judge": llm_choice}, span_output={"total": len(state["retrieved_chunks"]), "llm_judge_filtered": True, "dlq_saved": bool(state.get("dlq_saved"))})
                                    except Exception:
                                        pass
                                    return state
                        # If still unclear or no filter_path, fall through to clarify
                        state["needs_clarification"] = True
                        state["available_subspaces"] = list(cnt.keys())
                        state["clarifying_question"] = None  # will be filled by generate_clarification node
        except Exception as e:
            log.warning("Granularity check failed: %s", e)

    # --- Stage 3 (route): late fusion of the subspace ambiguity score ---
    # Gated by subspace_ambiguity_route and only when neither an explicit
    # filter_path nor a sticky/resolved subspace survived the existing logic —
    # explicit scope, sticky, and the general override above always win. Placed
    # after the granularity block so `applied_subspace` reflects what the
    # existing logic resolved (None = it could not decide).
    try:
        if (
            settings.subspace_ambiguity_route
            and not _explicit_scope
            and not state.get("applied_subspace")
            and _amb is not None
        ):
            if _amb.ambiguous:
                # Defer to the existing clarify gate (graph.should_clarify).
                state["needs_clarification"] = True
                if not state.get("available_subspaces"):
                    state["available_subspaces"] = list(SUBSPACES)
            elif _amb.subspace:
                # Confident subspace: clear the flag and route so the F5
                # filtered re-retrieve below runs against it.
                state["applied_subspace"] = _amb.subspace
                state["needs_clarification"] = False
    except Exception as _route_err:
        log.warning("subspace ambiguity routing failed: %s", _route_err)

    # --- Filtered re-retrieve (F5) ---
    # If scattered but we have an inferred/sticky subspace, re-retrieve filtered
    # Skip when explicit scope already used (authoritative)
    try:
        if not _explicit_scope and state.get("needs_clarification") is False and not granular:
            inferred = state.get("applied_subspace")
            if inferred:
                path_map = get_settings().subspace_map
                filter_path = path_map.get(inferred, "")
                if filter_path:
                    try:
                        async with KnowledgebaseClient() as client2:
                            fresult = await client2.retrieve(
                            search_query, top_k, request_id, filter_path=filter_path,
                            keyword_boost=_subspace_keyword_boost(inferred or filter_path),
                        )
                        if isinstance(fresult, dict):
                            f_rrf = fresult.get("rrf", [])
                            f_ce = fresult.get("ce", [])
                            f_injected = fresult.get("injected_entities", [])
                        else:
                            f_rrf = []
                            f_ce = list(fresult)
                            f_injected = []
                        f_merged: dict[str, dict] = {}
                        f_order: list[str] = []
                        for c in f_rrf:
                            d = _item_dict(c, "rrf")
                            key = d.get("chunk_id") or f"__rrf_{len(f_order)}"
                            if key not in f_merged:
                                f_merged[key] = d
                                f_order.append(key)
                        for c in f_ce:
                            d = _item_dict(c, "ce")
                            key = d.get("chunk_id") or f"__ce_{len(f_order)}"
                            if key in f_merged:
                                existing = f_merged[key]
                                existing["source"] = "both"
                                if existing.get("rank_ce") is None:
                                    existing["rank_ce"] = d.get("rank_ce")
                                if existing.get("rerank_score") is None:
                                    existing["rerank_score"] = d.get("rerank_score")
                                if existing.get("hybrid_score") is None:
                                    existing["hybrid_score"] = d.get("hybrid_score")
                                existing["score"] = (
                                    existing.get("rerank_score")
                                    if existing.get("rerank_score") is not None
                                    else d.get("score", existing.get("score"))
                                )
                                for k in ("document_id", "title", "heading", "content", "folder_hierarchy"):
                                    if not existing.get(k) and d.get(k):
                                        existing[k] = d[k]
                            else:
                                f_merged[key] = d
                                f_order.append(key)

                        def _f_best_rank(d: dict) -> int:
                            ranks = [r for r in (d.get("rank_rrf"), d.get("rank_ce")) if isinstance(r, int)]
                            return min(ranks) if ranks else 10**9

                        f_both = sorted(
                            (f_merged[k] for k in f_order if f_merged[k].get("source") == "both"),
                            key=_f_best_rank,
                        )
                        f_ce_only = sorted(
                            (f_merged[k] for k in f_order if f_merged[k].get("source") == "ce"),
                            key=lambda d: d.get("rank_ce") if isinstance(d.get("rank_ce"), int) else 10**9,
                        )
                        f_rrf_only = sorted(
                            (f_merged[k] for k in f_order if f_merged[k].get("source") == "rrf"),
                            key=lambda d: d.get("rank_rrf") if isinstance(d.get("rank_rrf"), int) else 10**9,
                        )
                        filtered_retrieved = (f_both + f_ce_only + f_rrf_only)[:20]
                        if state.get("faq_verified") and state.get("direct_faq_answer"):
                            filtered_retrieved.insert(0, {
                                "chunk_id": "__faq_direct__",
                                "document_id": "__faq_direct__",
                                "title": "FAQ Direct Answer",
                                "heading": "FAQ",
                                "content": state["direct_faq_answer"],
                                "score": 1.0,
                                "source": "faq_direct",
                                "rank_rrf": 0,
                                "rank_ce": 0,
                                "hybrid_score": 1.0,
                                "rerank_score": None,
                            })
                        state["retrieved_chunks"] = filtered_retrieved
                        # Merge injected entities from filtered re-retrieve
                        if f_injected:
                            existing_injected = state.get("injected_entities") or []
                            seen = {e.get("list_name") for e in existing_injected}
                            merged_injected = list(existing_injected)
                            for ent in f_injected:
                                if ent.get("list_name") not in seen:
                                    merged_injected.append(ent)
                                    seen.add(ent.get("list_name"))
                            state["injected_entities"] = merged_injected
                            state["supplemental"] = merged_injected
                        # --- Dynamic broadening: one re-retrieval then not-enough-context + DLQ (F5) ---
                        try:
                            state["retrieve_attempts"] = int(state.get("retrieve_attempts") or 0) + 1
                        except Exception:
                            state["retrieve_attempts"] = 1
                        _f_top = _compute_top_score(state["retrieved_chunks"])
                        _f_thr = _get_ev_threshold(filter_path or inferred)
                        _f_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
                        _f_has_lex = _has_lexical_evidence_retrieve(search_query or query, state.get("retrieved_chunks"), filter_path or inferred)
                        if _f_has_lex:
                            log.info("T6 F5 lexical evidence keeps %d chunks despite low score %.4f < %.3f", len(state.get("retrieved_chunks") or []), _f_top, _f_thr)
                        if not state.get("faq_verified") and _f_top < _f_thr and not _f_has_sup and not _f_has_lex:
                            state["retrieved_chunks"] = []
                            state["dlq_saved"] = True
                            try:
                                from ..tracing import trace_span as _tsp4
                                _tsp4(request_id, "retrieve-dlq", span_input={"query": query, "search_query": search_query, "top_score": _f_top, "threshold": _f_thr, "stage": "f5_filtered", "retrieve_attempts": state.get("retrieve_attempts")}, span_output={"dlq": True, "query": query, "cleared": True})
                            except Exception:
                                pass
                            try:
                                import json as _json4, time as _time4
                                rec4 = {"ts": _time4.time(), "request_id": request_id, "query": query, "search_query": search_query, "top_score": _f_top, "threshold": _f_thr, "stage": "f5_filtered", "dlq": True, "retrieve_attempts": state.get("retrieve_attempts")}
                                with open(_DLQ_PATH, "a", encoding="utf-8") as fh4:
                                    fh4.write(_json4.dumps(rec4, ensure_ascii=False) + "\n")
                            except Exception:
                                pass
                            log.info("F5 filtered re-retrieve still low (%.4f < %.3f) — cleared for refusal + DLQ", _f_top, _f_thr)
                        log.info("Filtered re-retrieve for subspace %s (%s): %d chunks", inferred, filter_path, len(filtered_retrieved) if not state.get("dlq_saved") else 0)
                    except Exception as e:
                        log.warning("Filtered re-retrieve failed: %s", e)
    except Exception as e:
        log.warning("Filtered re-retrieve check failed: %s", e)

    # --- T1 central fallback: if current result is still filtered & weak and no fallback yet, retry whole KB ---
    # Covers explicit, LLM-judge, F5, and any other filtered probe that left weak evidence before DLQ.
    try:
        if not state.get("fallback_used") and not state.get("faq_verified"):
            _cf_top = _compute_top_score(state.get("retrieved_chunks") or [])
            _cf_len = len(state.get("retrieved_chunks") or [])
            # T3: pre-compute with applied_subspace, but will recompute per _cf_active below if needed
            _cf_thr = _get_ev_threshold(state.get("applied_subspace"))
            _cf_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
            _cf_needs = (_cf_len == 0 or _cf_top < _cf_thr) and not _cf_has_sup
            if _cf_needs:
                # Determine if original probe was filtered (not whole KB)
                _cf_active = None
                if _explicit_scope and explicit_filter:
                    _cf_active = explicit_filter
                elif state.get("applied_subspace"):
                    # Map label to filter_path for audit
                    try:
                        _cf_active = get_settings().subspace_map.get(state.get("applied_subspace"), state.get("applied_subspace"))
                    except Exception:
                        _cf_active = str(state.get("applied_subspace"))
                # Also consider the last filtered probe that left weak evidence (F5 inferred)
                # If no active label but we have a filtered retrieval that was weak, treat as filtered
                # Heuristic: if applied_subspace is None but we were in filtered path earlier, _explicit_scope would be true.
                # Only fallback if we can identify a filtered origin; otherwise skip to avoid loop on whole-KB weak.
                if _cf_active:
                    await _t1_try_whole_kb_fallback(state, search_query, request_id, top_k=20, active_filter=_cf_active)
    except Exception as _cfe:
        log.warning("T1 central fallback failed: %s", _cfe)

    # --- T2: Broadened retry when whole-KB fallback still weak ---
    # If after T1 (or without T1) the retrieval is still weak (0 chunks or max < threshold),
    # do a second fallback with broadened params: whole KB, top_k=30, keyword_boost=1.5.
    # Server handles synonym beam (KB_SYNONYM_BEAM=5) — client just relaxes keyword bias.
    # This reuses KnowledgebaseClient (trust_env=False) via _t2_try_broadened_retry helper.
    # Audit: broaden_used, fallback_used (also set), stage_timing.retrieve_broadened.
    try:
        # Only consider broaden if not already broadened and still weak
        if not state.get("broaden_used") and not state.get("faq_verified"):
            _t2_top = _compute_top_score(state.get("retrieved_chunks") or [])
            _t2_thr = _get_ev_threshold(state.get("applied_subspace"))
            _t2_len = len(state.get("retrieved_chunks") or [])
            _t2_has_sup = bool(state.get("supplemental") or state.get("injected_entities"))
            _t2_needs = (_t2_len == 0 or _t2_top < _t2_thr) and not _t2_has_sup
            # Also need to know we already attempted whole-KB (either via T1 or original whole-KB)
            # We broaden even if T1 wasn't applicable (whole-KB already weak), but skip if we never had a filtered probe and original was already broadened? No — broaden is useful for whole-KB weak.
            # Guard: don't broaden if no chunks and we haven't tried whole-KB yet but filtered was weak with no fallback_used? Actually T1 would have fired; if T1 didn't fire because _cf_active was None (whole-KB weak), we still want to broaden here.
            if _t2_needs:
                await _t2_try_broadened_retry(state, search_query, request_id, top_k=30, keyword_boost=1.5)
    except Exception as _t2e:
        log.warning("T2 broadened fallback failed: %s", _t2e)

    # --- Final fallback: if we did one re-retrieval and still low with no supplement, DLQ (single-retry budget) ---
    # T1: don't clear fallback chunks even if still low (better than empty faith 0.0)
    try:
        if int(state.get("retrieve_attempts") or 0) >= 1 and not state.get("dlq_saved") and not state.get("faq_verified"):
            _final_top = _compute_top_score(state.get("retrieved_chunks"))
            _final_thr = _get_ev_threshold(state.get("applied_subspace"))
            _final_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
            _final_has_fallback = bool(state.get("fallback_used") and len(state.get("retrieved_chunks") or []) > 0)
            _final_has_lex = _has_lexical_evidence_retrieve(search_query or query, state.get("retrieved_chunks"), state.get("applied_subspace"))
            if _final_has_lex:
                log.info("T6 final lexical evidence keeps %d chunks despite low score %.4f < %.3f", len(state.get("retrieved_chunks") or []), _final_top, _final_thr)
            if _final_top < _final_thr and not _final_has_sup and not _final_has_fallback and not _final_has_lex:
                state["retrieved_chunks"] = []
                state["dlq_saved"] = True
                try:
                    from ..tracing import trace_span as _tsp_final
                    _tsp_final(request_id, "retrieve-dlq", span_input={"query": query, "search_query": search_query, "top_score": _final_top, "threshold": _final_thr, "stage": "final_fallback", "retrieve_attempts": state.get("retrieve_attempts")}, span_output={"dlq": True, "query": query, "cleared": True})
                except Exception:
                    pass
                try:
                    import json as _jsonF, time as _timeF
                    recF = {"ts": _timeF.time(), "request_id": request_id, "query": query, "search_query": search_query, "top_score": _final_top, "threshold": _final_thr, "stage": "final_fallback", "dlq": True, "retrieve_attempts": state.get("retrieve_attempts")}
                    with open(_DLQ_PATH, "a", encoding="utf-8") as fhF:
                        fhF.write(_jsonF.dumps(recF, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                log.info("Final DLQ fallback for request %s top=%.4f < %.3f", request_id, _final_top, _final_thr)
    except Exception:
        pass

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    if "stage_timing_ms" not in state or state["stage_timing_ms"] is None:
        state["stage_timing_ms"] = {}
    state["stage_timing_ms"]["retrieve"] = elapsed

    return state