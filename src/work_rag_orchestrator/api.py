"""FastAPI application for Orchestrator service."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse

from .config import get_settings
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    HealthResponse,
    ReadyResponse,
    DependencyStatus,
    RAGMetadata,
    Citation,
)
from .graph import rag_graph
from .state import RAGState

log = logging.getLogger(__name__)

# Dependency readiness cache
_deps_ready: dict[str, bool] = {"kb": False, "guardrails": False}


async def check_dependencies() -> list[DependencyStatus]:
    """Check readiness of all dependencies."""
    from .clients.guardrails import GuardrailsClient
    from .clients.knowledgebase import KnowledgebaseClient
    
    settings = get_settings()
    deps = []
    
    # Check KB
    async with KnowledgebaseClient() as kb_client:
        kb_ready = await kb_client.health_check()
    deps.append(DependencyStatus(
        name="knowledgebase",
        status="ready" if kb_ready else "not_ready",
        url=settings.kb_base_url,
        error=None if kb_ready else "Health check failed",
    ))
    
    # Check Guardrails
    async with GuardrailsClient() as gr_client:
        gr_ready = await gr_client.health_check()
        if gr_ready:
            ready_info = await gr_client.readiness_check()
            gr_ready = ready_info.get("status") == "ready"
    deps.append(DependencyStatus(
        name="guardrails",
        status="ready" if gr_ready else "not_ready",
        url=settings.guardrails_base_url,
        error=None if gr_ready else "Health/readiness check failed",
    ))
    
    return deps


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    log.info("Starting Orchestrator service...")
    yield
    log.info("Shutting down Orchestrator service...")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Work RAG Orchestrator",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    async def ready():
        deps = await check_dependencies()
        all_ready = all(d.status == "ready" for d in deps)
        return ReadyResponse(
            status="ready" if all_ready else "not_ready",
            dependencies=deps,
        )

    @app.get("/v1/models")
    async def list_models():
        # Open WebUI discovery — single RAG Agent (self-hosted)
        # Hides internal model IDs; WebUI shows only this name
        return {
            "object": "list",
            "data": [
                {"id": "work-rag-agent", "object": "model", "created": 0, "owned_by": "vast", "name": "Work RAG Agent"},
                {"id": "gemma-4-31b", "object": "model", "created": 0, "owned_by": "vast", "name": "Work RAG Agent"},
            ],
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
        x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
        authorization: Optional[str] = Header(None),
    ):
        # Generate request ID for tracing
        request_id = x_request_id or str(uuid.uuid4())
        t0 = time.monotonic()
        # Langfuse trace (self-hosted) — non-blocking, direct HTTP fallback
        try:
            from .tracing import start_trace
            input_text = request.messages[-1].get("content", "")[:1000] if request.messages else ""
            start_trace(request_id, "rag", input_text, {"model": request.model, "request_id": request_id})
        except Exception as e:
            log.warning("Langfuse trace init failed: %s", e)
        
        # Validate model — accept RAG Agent alias + Vast Gemma
        allowed_models = {"work-rag-agent", "gemma-4-31b", "unsloth/gemma-4-31B-it-GGUF", "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL", get_settings().upstream_llm_model}
        if request.model not in allowed_models:
            log.warning("Unsupported model requested: %s", request.model)
        
        # Prepare initial state
        initial_state: RAGState = {
            "request_id": request_id,
            "messages": request.messages,
            "query": "",
            "guardrail_decision": None,
            "retrieved_chunks": [],
            "prompt_messages": [],
            "answer": "",
            "citations": [],
            "error": None,
            "blocked": False,
            "refusal_message": None,
            "formatted_response": None,
            "retrieve_attempts": 0,
            "graded_chunks": [],
            "hallucination_spans": [],
            "grounded": False,
        }
        
        # Run the graph
        try:
            config = {"configurable": {"thread_id": request_id}}
            final_state = await rag_graph.ainvoke(initial_state, config=config)
        except Exception as e:
            log.exception("Graph execution failed: %s", e)
            raise HTTPException(status_code=500, detail="Orchestration failed")
        
        # Extract formatted response
        formatted = final_state.get("formatted_response", {})
        content = formatted.get("content", "An error occurred.")
        finish_reason = formatted.get("finish_reason", "error")
        citations = formatted.get("citations", [])
        
        # Build audit trail for transparency (shown in OpenWebUI rag agent)
        # This exposes the whole pipeline: guardrails, retrieval, generation
        try:
            from .schemas import AuditTrail
            # Get raw model output and guardrail decisions from state
            raw_output = final_state.get("answer", "") or final_state.get("refusal_message", "") or content
            # Build context string (first 2000 chars)
            prompt_msgs = final_state.get("prompt_messages", [])
            context_str = ""
            if prompt_msgs and len(prompt_msgs) > 1:
                # The user message contains the context + question
                context_str = prompt_msgs[1].get("content", "")[:2000]
            # Get retrieved chunks with scores
            retrieved = final_state.get("retrieved_chunks", [])
            # Build audit
            audit = AuditTrail(
                request_id=request_id,
                query=final_state.get("query", ""),
                guardrail_input=final_state.get("guardrail_decision"),
                retrieved_chunks=[
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "title": c.get("title", ""),
                        "heading": c.get("heading", ""),
                        "content": c.get("content", "")[:500],
                        "score": c.get("score", 0),
                    }
                    for c in retrieved[:5]
                ],
                reranker_scores=[c.get("score", 0) for c in retrieved[:5]],
                context_sent_to_gemma=context_str,
                raw_model_output=raw_output[:2000],
                guardrail_output={"blocked": final_state.get("blocked", False), "refusal": final_state.get("refusal_message")},
                final_answer=content,
                citations=[Citation(**c) for c in citations],
                model=request.model,
            )
        except Exception as e:
            log.warning(f"Failed to build audit trail: {e}")
            audit = None
        
        # Build response
        latency_ms = int((time.monotonic() - t0) * 1000)
        # Populate audit latency
        if audit is not None:
            try:
                audit.latency_ms = latency_ms
            except Exception:
                pass
        # Update Langfuse trace
        try:
            from .tracing import update_trace
            retrieved = final_state.get("retrieved_chunks", [])
            update_trace(
                request_id,
                content[:2000],
                {
                    "finish_reason": finish_reason,
                    "citations": len(citations),
                    "retrieved": len(retrieved),
                    "blocked": final_state.get("blocked", False),
                    "latency_ms": latency_ms,
                },
            )
        except Exception as e:
            log.warning("Langfuse trace update failed: %s", e)

        response = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": content},
                    finish_reason=finish_reason,
                )
            ],
            rag=RAGMetadata(
                request_id=request_id,
                citations=[Citation(**c) for c in citations],
            ),
            audit=audit,
        )
        
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


def main():
    """Entry point for running the service."""
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "work_rag_orchestrator.api:create_app",
        factory=True,
        host=settings.orchestrator_host,
        port=settings.orchestrator_port,
        log_level="info",
    )




app = create_app()

if __name__ == "__main__":
    main()