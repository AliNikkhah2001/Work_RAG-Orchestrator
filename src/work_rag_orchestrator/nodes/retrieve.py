"""Retrieve node - calls KB search API."""

from __future__ import annotations

import logging
from ..state import RAGState
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
    t = (text or "").strip().lower()
    for ch in "؟?!.,،؛:؛\"'«»()…-":
        t = t.replace(ch, "")
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


async def retrieve(state: RAGState) -> RAGState:
    """
    Rewrite the last user message against history, then call KB search.

    The standalone rewritten query is normalized and used for KB search;
    the original latest user message stays in state["query"] for audit,
    the rewritten form is stored in state["rewritten_query"].
    """
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
        log.info("Greeting-only (%s) for request %s — skipping retrieval", kind, request_id)
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
        return state

    # Client returns {"rrf": [...top10...], "ce": [...top10...]}; be
    # defensive if an older client shape (plain list) is ever returned.
    if isinstance(result, dict):
        rrf_chunks = result.get("rrf", [])
        ce_chunks = result.get("ce", [])
    else:  # pragma: no cover - legacy fallback
        rrf_chunks = []
        ce_chunks = list(result)

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
            for k in ("document_id", "title", "heading", "content"):
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

    return state