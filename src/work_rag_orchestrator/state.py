"""LangGraph state definition for RAG orchestration."""

from __future__ import annotations

from typing import TypedDict, List, Optional, Dict, Any
from uuid import UUID


class RAGState(TypedDict):
    """State carried through the RAG graph."""
    
    # Request identification
    request_id: str
    
    # Original messages from user
    messages: List[Dict[str, str]]
    
    # Extracted query (latest user message)
    query: str
    
    # Guardrails input check decision
    guardrail_decision: Optional[Dict[str, Any]]
    
    # Retrieved chunks from KB
    retrieved_chunks: List[Dict[str, Any]]
    
    # Built prompt messages for generation
    prompt_messages: List[Dict[str, str]]
    
    # Generated answer
    answer: str
    
    # Citations for the answer
    citations: List[Dict[str, Any]]
    
    # Error if any
    error: Optional[str]
    
    # Whether input was blocked by guardrails
    blocked: bool
    
    # Refusal message if blocked
    refusal_message: Optional[str]
    
    # Final OpenAI-shaped payload assembled by format_response
    # (content, finish_reason, citations) - consumed by the API layer.
    formatted_response: Optional[Dict[str, Any]]
    
    # Retrieval rewrite/grade loop counters (Phase 3+; unused in MVP)
    retrieve_attempts: int
    graded_chunks: List[Dict[str, Any]]
    hallucination_spans: List[str]
    grounded: bool