"""Retrieve node - calls KB search API."""

from __future__ import annotations

import logging
import time
from ..state import RAGState, record_timing
from ..config import get_settings
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
    """Best rerank/hybrid/score across chunks, 0.0 if none."""
    best = 0.0
    for c in (chunks or []):
        for k in ("rerank_score", "hybrid_score", "score"):
            v = c.get(k)
            if isinstance(v, (int, float)):
                if float(v) > best:
                    best = float(v)
                break
    return best


def _get_ev_threshold() -> float:
    try:
        from ..config import get_settings as _gs
        return float(_gs().evidence_min_score)
    except Exception:
        return 0.15


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
    settings = get_settings()
    top_k = settings.retrieval_top_k

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
                result = await client.retrieve(search_query, top_k, request_id, filter_path=explicit_filter)
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
                result = await client.retrieve(search_query, top_k, request_id)
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

    # --- G5 supplement fallback: if top evidence < threshold, try direct
    # EntityList content search (threshold 0.4 < 0.6) and log outcome ---
    try:
        from ..config import get_settings as _get_ev_settings
        _ev_threshold = float(_get_ev_settings().evidence_min_score)
    except Exception:
        _ev_threshold = 0.15
    try:
        _top_score = 0.0
        for _c in retrieved:
            for _k in ("rerank_score", "hybrid_score", "score"):
                _v = _c.get(_k)
                if isinstance(_v, (int, float)):
                    _top_score = max(_top_score, float(_v))
                    break
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
    if not state.get("faq_verified") and locals().get("_top_score", 1.0) < locals().get("_ev_threshold", 0.15) and not state.get("supplemental") and not state.get("injected_entities"):
        state["retrieved_chunks"] = []
        log.info("Low evidence still < %.3f after supplement for %s — clear for refusal", locals().get("_ev_threshold", 0.15), request_id)

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
                                _fresult_sticky = await _client_sticky.retrieve(
                                    search_query, top_k, request_id, filter_path=_filter_path_sticky
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
                            # --- Dynamic broadening: one re-retrieval then not-enough-context + DLQ ---
                            # Increment attempts (sticky counts as the single allowed re-retrieve)
                            try:
                                state["retrieve_attempts"] = int(state.get("retrieve_attempts") or 0) + 1
                            except Exception:
                                state["retrieve_attempts"] = 1
                            _sticky_top = _compute_top_score(state["retrieved_chunks"])
                            _sticky_thr = _get_ev_threshold()
                            _sticky_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
                            if not state.get("faq_verified") and _sticky_top < _sticky_thr and not _sticky_has_sup:
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
                granular = (is_general and _general_exempt and _is_general_query) or (n / total >= _thresh)
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
                        if llm_choice and llm_choice in ("cheque","facilities_person","facilities_legal","general"):
                            filter_path = get_settings().subspace_map.get(llm_choice, "")
                            if filter_path:
                                _fresult_llm = None
                                try:
                                    async with KnowledgebaseClient() as _client_llm:
                                        _fresult_llm = await _client_llm.retrieve(search_query, top_k, request_id, filter_path=filter_path)
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
                                    _llm_thr = _get_ev_threshold()
                                    _llm_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
                                    if not state.get("faq_verified") and _llm_top < _llm_thr and not _llm_has_sup:
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
                            fresult = await client2.retrieve(search_query, top_k, request_id, filter_path=filter_path)
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
                        _f_thr = _get_ev_threshold()
                        _f_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
                        if not state.get("faq_verified") and _f_top < _f_thr and not _f_has_sup:
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

    # --- Final fallback: if we did one re-retrieval and still low with no supplement, DLQ (single-retry budget) ---
    try:
        if int(state.get("retrieve_attempts") or 0) >= 1 and not state.get("dlq_saved") and not state.get("faq_verified"):
            _final_top = _compute_top_score(state.get("retrieved_chunks"))
            _final_thr = _get_ev_threshold()
            _final_has_sup = bool(state.get("injected_entities") or state.get("supplemental"))
            if _final_top < _final_thr and not _final_has_sup:
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