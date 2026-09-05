"""Pydantic schemas for Orchestrator API."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(..., description="Model identifier")
    messages: List[Dict[str, str]] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    session_id: Optional[str] = Field(default=None)


class Citation(BaseModel):
    """Citation metadata."""

    chunk_id: str
    document_id: str
    title: str
    heading: str


class ChatCompletionChoice(BaseModel):
    """Single choice in chat completion response."""

    index: int
    message: Dict[str, str]
    finish_reason: str


class RAGMetadata(BaseModel):
    """RAG-specific metadata in response."""

    request_id: str
    citations: List[Citation] = Field(default_factory=list)


class AuditTrail(BaseModel):
    """Full audit trail for transparency — shown in OpenWebUI rag agent."""

    request_id: str
    query: str
    guardrail_input: Optional[Dict[str, Any]] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    reranker_scores: List[float] = Field(default_factory=list)
    context_sent_to_gemma: str = ""
    raw_model_output: str = ""
    guardrail_output: Optional[Dict[str, Any]] = None
    final_answer: str = ""
    citations: List[Citation] = Field(default_factory=list)
    latency_ms: Optional[Dict[str, float]] = None
    model: str = ""


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response with RAG metadata + audit."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex[:29]}")
    object: str = "chat.completion"
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict[str, int]] = None
    rag: Optional[RAGMetadata] = None
    audit: Optional[AuditTrail] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"


class DependencyStatus(BaseModel):
    """Individual dependency status."""

    name: str
    status: str  # "ready", "not_ready", "error"
    url: str
    error: Optional[str] = None


class ReadyResponse(BaseModel):
    """Readiness check response."""

    status: str  # "ready" or "not_ready"
    dependencies: List[DependencyStatus]


class RailCheckRequest(BaseModel):
    """Request for guardrails check."""

    stage: str
    text: str
    request_id: str


class RailCheckResponse(BaseModel):
    """Response from guardrails check."""

    allowed: bool
    action: str
    categories: List[str] = Field(default_factory=list)
    reason: Optional[str] = None
    policy_version: str
    request_id: str


class KBRetrievalRequest(BaseModel):
    """KB retrieval request."""

    query: str
    top_k: int


class KBRetrievalResult(BaseModel):
    """Single retrieval result."""

    chunk_id: str
    document_id: str
    title: str
    heading: str
    content: str
    score: float


class KBRetrievalResponse(BaseModel):
    """KB retrieval response."""

    final_results: List[KBRetrievalResult]


class GuardrailsChatRequest(BaseModel):
    """Guardrails chat completion request."""

    model: str
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GuardrailsChatResponse(BaseModel):
    """Guardrails chat completion response."""

    choices: List[ChatCompletionChoice]
    model: str