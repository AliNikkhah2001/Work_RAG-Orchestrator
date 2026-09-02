"""FastAPI application for Orchestrator service."""

from __future__ import annotations

import logging
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

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
        x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
        authorization: Optional[str] = Header(None),
    ):
        # Generate request ID for tracing
        request_id = x_request_id or str(uuid.uuid4())
        
        # Validate model — accept Vast Gemma and legacy alias
        allowed_models = {"gemma-4-31b", "unsloth/gemma-4-31B-it-GGUF", "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL", get_settings().upstream_llm_model}
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
        
        # Build response
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
        host="127.0.0.1",
        port=settings.orchestrator_port,
        log_level="info",
    )


app = create_app()

if __name__ == "__main__":
    main()