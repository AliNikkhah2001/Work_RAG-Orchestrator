"""Retrieve node - calls KB search API."""

from __future__ import annotations

import logging
from ..state import RAGState
from ..config import get_settings
from ..clients.knowledgebase import KnowledgebaseClient
from ..rewrite import rewrite_query, last_exchanges_text

log = logging.getLogger(__name__)


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
    
    async with KnowledgebaseClient() as client:
        chunks = await client.retrieve(search_query, top_k, request_id)
    
    # Normalize to dict format for state
    state["retrieved_chunks"] = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "title": c.title,
            "heading": c.heading,
            "content": c.content,
            "score": c.score,
        }
        for c in chunks
    ]
    
    log.info("Retrieved %d chunks for request %s", len(chunks), request_id)

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
            span_output=[
                {"chunk_id": c.get("chunk_id"), "title": c.get("title"), "score": c.get("score")}
                for c in state["retrieved_chunks"][:5]
            ],
        )
    except Exception:
        pass

    return state