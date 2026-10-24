"""Pydantic schemas for Orchestrator API."""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Model identifier")
    messages: List[Dict[str, str]] = Field(..., min_length=1)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    stream: bool = Field(default=False)
    session_id: Optional[str] = Field(default=None)
    filter_path: Optional[str] = Field(default=None, description="KB subspace filter path, e.g. 'اشخاص حقوقی' or 'گزارش اعتباری چک'")


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
    supplemental: Optional[List[Dict[str, Any]]] = None
    follow_ups: Optional[List[str]] = Field(default=None, description="Grounded follow-up questions answerable from same retrieved chunks")


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
    latency_ms: Optional[Any] = None
    model: str = ""
    stage_timing_ms: Optional[Dict[str, float]] = None
    original_query: str = ""
    rewritten_query: str = ""
    guardrail_input_signals: List[Dict[str, Any]] = Field(default_factory=list)
    guardrail_output_signals: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_method: str = ""
    retrieval_chunks_total: int = 0
    retrieval_chunks_used: int = 0
    reranker_applied: bool = False
    context_preview: str = ""
    prompt_messages_preview: List[Dict[str, str]] = Field(default_factory=list)
    supplemental: Optional[List[Dict[str, Any]]] = None
    applied_subspace: Optional[str] = Field(default=None, description="Subspace applied for retrieval (cheque/facilities/general/legal_entity)")
    available_subspaces: Optional[List[str]] = None
    # T1 fallback audit
    fallback_used: bool = Field(default=False, description="Whether filtered→whole-KB fallback was triggered")
    fallback_reason: Optional[str] = Field(default=None, description="Reason for fallback (empty/low_score) or retry failure")
    retry_stage_timing: Optional[Dict[str, float]] = Field(default=None, description="Timing for retrieve_retry/broadened stages")
    broaden_used: bool = Field(default=False, description="Whether broadened retry (top_k 30, boost 1.5) was triggered")
    # Embedding subspace ambiguity score (docs/SUBSPACE_AMBIGUITY.md, stage 1 instrument)
    ambiguity_score: Optional[float] = Field(default=None, description="Subspace ambiguity in [0,1] (higher = more ambiguous)")
    ambiguity_margin: Optional[float] = Field(default=None, description="Top1-top2 subspace similarity margin")
    ambiguity_entropy: Optional[float] = Field(default=None, description="Normalized softmax entropy of subspace sims in [0,1]")
    ambiguity_ambiguous: Optional[bool] = Field(default=None, description="Whether the ambiguity thresholds flagged the query")
    ambiguity_source: Optional[str] = Field(default=None, description="Scorer source: protos|classifier|fallback|disabled")
    subspace_sims: Optional[Dict[str, float]] = Field(default=None, description="Per-subspace cosine similarity map (best prototype)")
    ambiguity_intra_spread: Optional[Dict[str, float]] = Field(default=None, description="Per-subspace std over prototype similarities")
    ambiguity_proto_coverage: Optional[Dict[str, float]] = Field(default=None, description="Per-subspace fraction of prototypes above coverage threshold")


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
    source: Optional[str] = None
    rank_rrf: Optional[int] = None
    rank_ce: Optional[int] = None
    hybrid_score: Optional[float] = None
    rerank_score: Optional[float] = None
    folder_hierarchy: Optional[List[str]] = None


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