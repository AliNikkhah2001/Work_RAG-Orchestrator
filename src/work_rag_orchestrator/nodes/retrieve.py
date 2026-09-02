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
    settings = get_settings()
    top_k = settings.retrieval_top_k
    
    async with KnowledgebaseClient() as client:
        chunks = await client.retrieve(query, top_k, request_id)
    
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
    return state