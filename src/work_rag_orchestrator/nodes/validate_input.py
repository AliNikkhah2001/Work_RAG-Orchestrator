"""Validate input node - local signal detection only (no LLM call).

Extracts latest user message and runs basic local checks.
Does NOT call TRT-LLM/Gemma - that happens once in guarded_generate.
"""
from __future__ import annotations

import logging
import time
from ..state import RAGState, record_timing

log = logging.getLogger(__name__)


async def validate_input(state: RAGState) -> RAGState:
    """
    Extract latest user message and run local signal detection only.

    If blocked, set blocked=True and refusal_message.
    If allowed, continue to retrieve node.

    Does NOT call TRT-LLM/Gemma - that happens once in guarded_generate.
    """
    t0 = time.monotonic()
    request_id = state["request_id"]
    messages = state["messages"]

    # Extract latest user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        state["error"] = "No user message found"
        state["blocked"] = True
        state["refusal_message"] = "No user message in request"
        elapsed = round((time.monotonic() - t0) * 1000, 1)
        record_timing(state, "validate_input", elapsed)
        return state

    query = user_messages[-1]["content"]
    state["query"] = query
    log.info("validate_input query=%r request_id=%s", query[:200], request_id)

    # Local signal detection only - no LLM call
    blocked = False
    refusal_reason = None

    low = query.strip().lower()
    if not low:
        blocked = True
        refusal_reason = "Empty query"

    state["blocked"] = blocked
    state["refusal_message"] = refusal_reason

    if blocked:
        log.info("Input blocked locally: %s", refusal_reason)

    elapsed = round((time.monotonic() - t0) * 1000, 1)
    if "stage_timing_ms" not in state or state["stage_timing_ms"] is None:
        state["stage_timing_ms"] = {}
    state["stage_timing_ms"]["validate_input"] = elapsed

    return state
