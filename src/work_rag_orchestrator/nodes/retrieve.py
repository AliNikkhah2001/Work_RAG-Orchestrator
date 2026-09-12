"""Retrieve node - calls KB search API."""

from __future__ import annotations

import logging
from ..state import RAGState
from ..config import get_settings
from ..clients.knowledgebase import KnowledgebaseClient

log = logging.getLogger(__name__)


async def retrieve(state: RAGState) -> RAGState:
    """
    Call KB POST /search/api with top_k=5, normalize final_results.
    
    Uses the query from state["query"] (latest user message).
    Stores normalized chunks in state["retrieved_chunks"].
    """
    request_id = state["request_id"]
    query = state["query"]
    # Normalize informal Persian: چیه -> چیست for KB search (improves BM25 on short definition queries)
    norm_query = query.replace("چیه", "چیست").replace("چيه", "چیست")
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
            span_input={"query": query, "search_query": search_query, "top_k": top_k},
            span_output=[
                {"chunk_id": c.get("chunk_id"), "title": c.get("title"), "score": c.get("score")}
                for c in state["retrieved_chunks"][:5]
            ],
        )
    except Exception:
        pass

    return state