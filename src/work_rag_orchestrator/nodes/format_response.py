"""Format response node - returns OpenAI-compatible output with citation metadata."""

from __future__ import annotations

import logging
from ..state import RAGState
from ..schemas import Citation

log = logging.getLogger(__name__)


async def format_response(state: RAGState) -> RAGState:
    """
    Return OpenAI-compatible output with citation metadata.
    
    If blocked, returns refusal with content_filter finish_reason.
    If allowed, returns answer with citations from retrieved chunks.
    """
    request_id = state["request_id"]
    answer = state["answer"]
    blocked = state.get("blocked", False)
    refusal_message = state.get("refusal_message")
    chunks = state["retrieved_chunks"]
    
    if blocked:
        content = refusal_message or "I cannot comply with that request."
        finish_reason = "content_filter"
        citations = []
    else:
        content = answer
        finish_reason = "stop"
        # Build citations from retrieved chunks
        citations = [
            Citation(
                chunk_id=c.get("chunk_id", ""),
                document_id=c.get("document_id", ""),
                title=c.get("title", ""),
                heading=c.get("heading", ""),
            )
            for c in chunks
            if c.get("chunk_id")
        ]
    
    # Store formatted response data in state for API layer
    state["formatted_response"] = {
        "content": content,
        "finish_reason": finish_reason,
        "citations": [c.model_dump() for c in citations],
    }
    
    log.info("Formatted response for request %s", request_id)
    return state