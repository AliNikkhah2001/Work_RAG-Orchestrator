"""Mizan Retrieval Judge node - post-cross-encoder relevance & subspace validation.

This node runs after the retrieve node (after cross-encoder reranking) and uses
Mizan System-1 to judge:
1. Relevance of retrieved chunks to the query
2. Whether chunks match the applied subspace
3. Whether evidence is sufficient or need to retrieve again
4. Whether subspace should change for the new query

Actions: accept | reject | retrieve_again | change_subspace
"""

from __future__ import annotations

import logging
import time
from ..state import RAGState, record_timing
from ..clients.mizan import mizan_retrieval_judge, mizan_subspace_change_decision
from ..config import get_settings

log = logging.getLogger(__name__)


async def mizan_retrieval_judge_node(state: RAGState) -> RAGState:
    """
    Mizan judges the retrieved chunks after cross-encoder reranking.
    
    This is a calibrated Persian decision model that evaluates:
    - Topical relevance of chunks to query
    - Subspace alignment (do chunks match applied_subspace?)
    - Evidence sufficiency
    - Quality scoring (0-10)
    
    Based on Mizan's decision, we either:
    - Accept: keep chunks, continue to build_context
    - Reject: clear chunks, trigger refusal (G5)
    - Retrieve_again: broaden search, retry retrieval
    - Change_subspace: switch subspace, retry retrieval
    """
    t0 = time.monotonic()
    request_id = state["request_id"]
    query = state.get("query", "")
    rewritten_query = state.get("rewritten_query", query)
    chunks = state.get("retrieved_chunks", [])
    applied_subspace = state.get("applied_subspace")

    # Skip if no retrieval happened (greeting, spec, FAQ)
    if state.get("greeting_only") or state.get("classify") == "specification" or state.get("faq_verified"):
        record_timing(state, "mizan_retrieval_judge", round((time.monotonic() - t0) * 1000, 1))
        return state

    # Skip if already in fallback/broadened path (avoid infinite loops)
    if state.get("mizan_judge_applied"):
        record_timing(state, "mizan_retrieval_judge", round((time.monotonic() - t0) * 1000, 1))
        return state

    try:
        # Get Mizan's judgment on retrieval quality
        judgment = await mizan_retrieval_judge(
            query=query,
            rewritten_query=rewritten_query,
            chunks=chunks,
            applied_subspace=applied_subspace,
            request_id=request_id,
        )
    except Exception as e:
        log.warning("Mizan retrieval judge failed for request %s: %s", request_id, e)
        # Fail open - continue with original chunks
        state["mizan_judge_error"] = str(e)
        record_timing(state, "mizan_retrieval_judge", round((time.monotonic() - t0) * 1000, 1))
        return state

    action = judgment.get("action", "accept")
    confidence = judgment.get("confidence", 0.0)
    reasoning = judgment.get("reasoning", "")
    keep_chunks = judgment.get("keep_chunks", [])
    new_subspace = judgment.get("new_subspace")
    retry_params = judgment.get("retrieve_again_params")

    # Store judgment for audit
    state["mizan_retrieval_judgment"] = judgment
    state["mizan_judge_applied"] = True

    log.info(
        "Mizan retrieval judge for request %s: action=%s conf=%.2f reasoning=%s",
        request_id, action, confidence, reasoning[:100]
    )

    # Execute action
    if action == "accept":
        # Keep only the chunks Mizan approved
        if keep_chunks:
            state["retrieved_chunks"] = [c for c in chunks if c.get("chunk_id") in keep_chunks]
        state["mizan_accepted"] = True
        state["mizan_confidence"] = confidence

    elif action == "reject":
        # Clear chunks - will trigger G5 no-evidence guard in format_response
        state["retrieved_chunks"] = []
        state["mizan_rejected"] = True
        state["mizan_reject_reason"] = reasoning

    elif action == "retrieve_again":
        # Mark for re-retrieval with broadened parameters
        state["mizan_retrieve_again"] = True
        state["mizan_retry_params"] = retry_params or {"reason": "mizan_insufficient", "broaden": True}
        # Keep current chunks as fallback but mark for replacement
        state["mizan_pre_retry_chunks"] = chunks.copy()

    elif action == "change_subspace":
        # Subspace mismatch detected - will re-retrieve with new subspace
        state["mizan_change_subspace"] = True
        state["mizan_new_subspace"] = new_subspace
        state["mizan_pre_retry_chunks"] = chunks.copy()

    record_timing(state, "mizan_retrieval_judge", round((time.monotonic() - t0) * 1000, 1))
    return state


async def mizan_subspace_continuity_node(state: RAGState) -> RAGState:
    """
    Mizan decides if subspace should change for a new query in multi-turn conversation.
    
    Runs after classify but before retrieve, when we have a sticky subspace from
    previous turns. Uses Mizan to evaluate if the new query warrants a subspace change.
    """
    t0 = time.monotonic()
    request_id = state["request_id"]
    query = state.get("query", "")
    history = state.get("messages", [])
    current_subspace = state.get("applied_subspace")

    # Only run if we have a sticky subspace and this is a question (not greeting/spec)
    if not current_subspace or state.get("classify") in ("greeting", "specification"):
        record_timing(state, "mizan_subspace_continuity", round((time.monotonic() - t0) * 1000, 1))
        return state

    # Build history text
    history_text = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}" 
        for m in history[-6:]  # Last 3 exchanges
    )

    try:
        new_subspace, confidence, reasoning = await mizan_subspace_change_decision(
            query=query,
            history=history_text,
            current_subspace=current_subspace,
            request_id=request_id,
        )
    except Exception as e:
        log.warning("Mizan subspace continuity failed for request %s: %s", request_id, e)
        record_timing(state, "mizan_subspace_continuity", round((time.monotonic() - t0) * 1000, 1))
        return state

    # If subspace changed, update applied_subspace
    if new_subspace != current_subspace:
        log.info(
            "Mizan subspace change for request %s: %s -> %s (conf=%.2f) reasoning=%s",
            request_id, current_subspace, new_subspace, confidence, reasoning[:100]
        )
        state["applied_subspace"] = new_subspace
        state["mizan_subspace_changed"] = True
        state["mizan_subspace_confidence"] = confidence
        state["mizan_subspace_reason"] = reasoning
        # Clear sticky decay counter since explicit change
        state["sticky_generic_turns"] = 0
    else:
        state["mizan_subspace_kept"] = True
        state["mizan_subspace_confidence"] = confidence

    record_timing(state, "mizan_subspace_continuity", round((time.monotonic() - t0) * 1000, 1))
    return state