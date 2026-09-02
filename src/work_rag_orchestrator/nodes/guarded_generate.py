"""Guarded generate node - calls Guardrails chat completion (which calls Gemma)."""

from __future__ import annotations

import logging
from ..state import RAGState
from ..clients.guardrails import GuardrailsClient
from ..schemas import GuardrailsChatRequest
from ..config import get_settings

log = logging.getLogger(__name__)


async def guarded_generate(state: RAGState) -> RAGState:
    """
    Call Guardrails POST /v1/chat/completions, which calls Gemma.
    
    Uses prompt_messages from state. Stores answer in state["answer"].
    """
    request_id = state["request_id"]
    prompt_messages = state["prompt_messages"]
    settings = get_settings()
    
    # Prepare request for Guardrails
    # Guardrails will enforce output rails and call upstream Gemma
    # Use configured Vast Gemma model; preserve backwards compat with gemma-4-31b
    req_model = settings.upstream_llm_model
    # If state carries original model (future), prefer it; else config
    if state.get("messages"):
        orig = state["messages"][-1] if state["messages"] else {}
        # keep config as source of truth for Vast
        req_model = settings.upstream_llm_model
    request = GuardrailsChatRequest(
        model=req_model,
        messages=prompt_messages,
        max_tokens=512,
        temperature=0.2,
    )
    
    async with GuardrailsClient() as client:
        response = await client.chat_completion(request, request_id)
    
    # Extract answer
    if response.choices:
        answer = response.choices[0].message.get("content", "")
        finish_reason = response.choices[0].finish_reason
    else:
        answer = ""
        finish_reason = "error"
    
    # Check if generation was blocked by output rails
    if finish_reason == "content_filter":
        state["blocked"] = True
        state["refusal_message"] = answer
        state["answer"] = ""
    else:
        state["answer"] = answer
        state["blocked"] = False
    
    log.info("Generation completed for request %s (finish_reason: %s)", request_id, finish_reason)
    return state