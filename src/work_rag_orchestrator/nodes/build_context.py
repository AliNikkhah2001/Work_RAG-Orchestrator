"""Build context node - creates bounded system/context message with numbered source IDs."""

from __future__ import annotations

import logging
from ..state import RAGState

log = logging.getLogger(__name__)

# Max context size (roughly tokens * 4 for chars) — Vast: reduced for RTX 6000 + faster Gemma
MAX_CONTEXT_CHARS = 4000
MAX_CHUNKS = 3


async def build_context(state: RAGState) -> RAGState:
    """
    Create a bounded system/context message with numbered source IDs.
    
    Uses retrieved_chunks to build context. Limits total size and chunk count.
    Stores prompt_messages in state for the generation step.
    """
    request_id = state["request_id"]
    query = state["query"]
    chunks = state["retrieved_chunks"]
    
    # Limit chunks
    chunks = chunks[:MAX_CHUNKS]
    
    # Build context with numbered sources
    context_parts = ["[Context from Knowledge Base]"]
    
    for i, chunk in enumerate(chunks, 1):
        source_id = f"[{i}]"
        title = chunk.get("title", "Unknown")
        heading = chunk.get("heading", "")
        content = chunk.get("content", "")
        
        context_entry = f"{source_id} Title: {title}"
        if heading:
            context_entry += f"\nHeading: {heading}"
        context_entry += f"\nContent: {content}"
        
        context_parts.append(context_entry)
    
    context_text = "\n\n".join(context_parts)
    
    # Truncate if too long
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS] + "\n...[truncated]"
    
    # Build system message with instructions
    system_message = (
        "You are a helpful assistant for ICS Credit Scoring. "
        "Answer the user's question using ONLY the provided context. "
        "If the context doesn't contain enough information, say so explicitly. "
        "Cite sources using the numbered brackets [1], [2], etc. "
        "Do not make up information not in the context."
    )
    
    # Build prompt messages for generation
    prompt_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{context_text}\n\nQuestion: {query}"},
    ]
    
    state["prompt_messages"] = prompt_messages
    log.info("Built context with %d sources for request %s", len(chunks), request_id)
    return state