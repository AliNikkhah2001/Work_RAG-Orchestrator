"""Validate input node - calls Guardrails input rail check."""

from __future__ import annotations

import logging
from ..state import RAGState
from ..clients.guardrails import GuardrailsClient

log = logging.getLogger(__name__)


async def validate_input(state: RAGState) -> RAGState:
    """
    Extract latest user message and call Guardrails input rail check.
    
    If blocked, set blocked=True and refusal_message.
    If allowed, continue to retrieve node.
    """
    request_id = state["request_id"]
    messages = state["messages"]
    
    # Extract latest user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        state["error"] = "No user message found"
        state["blocked"] = True
        state["refusal_message"] = "No user message in request"
        return state
    
    query = user_messages[-1]["content"]
    state["query"] = query
    log.info("validate_input query=%r request_id=%s", query[:200], request_id)
    
    # Call Guardrails input check
    async with GuardrailsClient() as client:
        decision = await client.check_rails("input", query, request_id)
    log.info("validate_input decision allowed=%s categories=%s reason=%r", decision.allowed, decision.categories, decision.reason)
    
    state["guardrail_decision"] = decision.model_dump()
    
    if not decision.allowed:
        log.info("Input blocked by guardrails: %s", decision.reason)
        state["blocked"] = True
        state["refusal_message"] = decision.reason or "I cannot comply with that request."
    else:
        state["blocked"] = False
    
    return state