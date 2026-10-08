"""FastAPI application for Orchestrator service."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
import re

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
    from .clients.guardrails import _aclose_shared_client
    await _aclose_shared_client()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="ICS Helper Agent",
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
        # Open WebUI discovery — single ICS Helper Agent (self-hosted)
        # Exposes only ICS Helper Agent; internal aliases hidden from UI but still accepted for backward compat
        return {
            "object": "list",
            "data": [
                {"id": "ics-helper-agent", "object": "model", "created": 0, "owned_by": "ics", "name": "ICS Helper Agent"},
            ],
        }

    # /rate command pattern: /rate 5 comment #tag1 #tag2
    _RATE_RE = re.compile(r"^\s*/rate\s+([1-5])\s*(.*?)\s*(#[\S]+(?:\s+#[\S]+)*)?\s*$", re.IGNORECASE | re.DOTALL)
    _REPORT_RE = re.compile(r"^\s*/(?:report|problem)\s+(\w+)\s*(.*?)\s*(#[\S]+(?:\s+#[\S]+)*)?\s*$", re.IGNORECASE | re.DOTALL)
    _REPORT_CATEGORIES = {"inaccurate","incomplete","hallucination","wrong_source","off_topic","other"}

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(
        request: ChatCompletionRequest,
        x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
        authorization: Optional[str] = Header(None),
    ):
        # --- /rate command: save evaluation without calling LLM ---
        try:
            last_content = request.messages[-1].get("content", "") if request.messages else ""
            m = _RATE_RE.match(last_content.strip())
            if m:
                import httpx as _httpx
                rating = int(m.group(1))
                comment = (m.group(2) or "").strip()
                tags_raw = (m.group(3) or "").strip()
                tags = re.findall(r"#([^\s#]+)", tags_raw) if tags_raw else []
                # Use chat-level correlation: try X-Request-ID, else generate
                rate_request_id = x_request_id or str(uuid.uuid4())
                # Try to save to observability store (best-effort, fail-silent for chat)
                stars = "★" * rating + "☆" * (5 - rating)
                tag_str = f"  Tags: {', '.join('#'+t for t in tags)}" if tags else ""
                cmt_str = f"\n> {comment}" if comment else ""
                # Fire-and-forget POST to observability
                try:
                    import os as _os
                    obs_url = _os.getenv("OBSERVABILITY_URL", "http://127.0.0.1:3000")
                    # Also try Docker service name
                    for base in [obs_url, "http://rag-tracing-fallback:3000"]:
                        try:
                            async with _httpx.AsyncClient(timeout=3.0) as _client:
                                await _client.post(
                                    f"{base.rstrip('/')}/api/observability/evaluations",
                                    json={
                                        "request_id": rate_request_id,
                                        "rating": rating,
                                        "comment": comment,
                                        "tags": tags,
                                        "user_query": last_content[:500],
                                        "model": request.model,
                                    },
                                )
                            break
                        except Exception:
                            continue
                except Exception:
                    pass
                msg = (
                    f"Rating saved: {stars} ({rating}/5){cmt_str}{tag_str}\n\n"
                    f"View in dashboard: http://127.0.0.1:3000/dashboard/observability"
                )
                return ChatCompletionResponse(
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": msg}, finish_reason="stop")],
                    rag=RAGMetadata(request_id=rate_request_id, citations=[]),
                    audit=None,
                )
            m2 = _REPORT_RE.match(last_content.strip())
            if m2:
                import httpx as _httpx2
                category = (m2.group(1) or "").lower().strip()
                if category not in _REPORT_CATEGORIES:
                    category = "other"
                comment2 = (m2.group(2) or "").strip()
                tags_raw2 = (m2.group(3) or "").strip()
                tags2 = re.findall(r"#([^\s#]+)", tags_raw2) if tags_raw2 else []
                rate_request_id2 = x_request_id or str(uuid.uuid4())
                # Try to extract rating if combined "/rate N ... /report ..."
                rating_val = None
                try:
                    _rm = _RATE_RE.search(last_content)
                    if _rm:
                        rating_val = int(_rm.group(1))
                except Exception:
                    rating_val = None
                try:
                    import os as _os2
                    obs_url2 = _os2.getenv("OBSERVABILITY_URL", "http://127.0.0.1:3000")
                    for base in [obs_url2, "http://rag-tracing-fallback:3000"]:
                        try:
                            async with _httpx2.AsyncClient(timeout=3.0) as _client:
                                await _client.post(
                                    f"{base.rstrip('/')}/api/observability/evaluations",
                                    json={
                                        "request_id": rate_request_id2,
                                        "rating": rating_val,
                                        "comment": f"[REPORT:{category}] {comment2}".strip(),
                                        "tags": tags2 + [f"report:{category}"],
                                        "user_query": last_content[:500],
                                        "model": request.model,
                                    },
                                )
                            break
                        except Exception:
                            continue
                except Exception:
                    pass
                cmt_str2 = f"\n> {comment2}" if comment2 else ""
                tag_str2 = f"  Tags: {', '.join('#'+t for t in (tags2 + [f'report:{category}']))}"
                msg2 = (
                    f"Report saved: [{category}]{cmt_str2}{tag_str2}\n\n"
                    f"View in dashboard: http://127.0.0.1:3000/dashboard/observability"
                )
                return ChatCompletionResponse(
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": msg2}, finish_reason="stop")],
                    rag=RAGMetadata(request_id=rate_request_id2, citations=[]),
                    audit=None,
                )
        except Exception:
            pass  # fall through to normal RAG flow on any error

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
        
        # Validate model — primary is ICS Helper Agent; keep legacy aliases for backward compat
        allowed_models = {"ics-helper-agent", "work-rag-agent", "gemma-4-31b", "unsloth/gemma-4-31B-it-GGUF", "unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL", get_settings().upstream_llm_model}
        if request.model not in allowed_models:
            log.warning("Unsupported model requested: %s", request.model)
        
        # Prepare initial state
        initial_state: RAGState = {
            "request_id": request_id,
            "messages": request.messages,
            "query": "",
            "rewritten_query": "",
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
            "direct_faq_answer": None,
            "faq_matched": False,
            "greeting_only": None,
            "stage_timing_ms": {},
        }
        
        # --- Streaming branch: REAL token streaming (no artificial delay) ---
        # Multi-agent progress via SSE comments + OpenAI delta chunks.
        # Non-LLM nodes (validate/retrieve/build_context) run buffered (~300ms),
        # then LLM tokens are proxied as they arrive from Guardrails->Gemma (30-50ms/token).
        if request.stream:
            async def event_generator():
                # 1) Role preamble (required by OpenAI spec)
                preamble = {
                    "id": f"chatcmpl-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(preamble, ensure_ascii=False)}\n\n"
                try:
                    from .nodes.validate_input import validate_input as node_validate
                    from .nodes.retrieve import retrieve as node_retrieve
                    from .nodes.build_context import build_context as node_build
                    from .nodes.format_response import format_response as node_format
                    from .clients.guardrails import GuardrailsClient
                    from .schemas import GuardrailsChatRequest

                    # Validate + Retrieve in PARALLEL (query set upfront; retrieve doesn't need validate output)
                    # Saves ~30ms overlap since retrieve (11s) dwarfs check_rails (30ms).
                    query = initial_state["messages"][-1]["content"] if initial_state.get("messages") else ""
                    initial_state["query"] = query
                    yield ": agent validate_input + retrieve running\n\n"
                    t_retrieve = time.monotonic()
                    # O-C2: clone state to avoid race on shared dict
                    v_state = dict(initial_state)
                    r_state = dict(initial_state)
                    v_task = asyncio.create_task(node_validate(v_state))
                    r_task = asyncio.create_task(node_retrieve(r_state))
                    state = await v_task
                    if state.get("blocked"):
                        r_task.cancel()
                        try:
                            await r_task
                        except asyncio.CancelledError:
                            pass
                        # merge non-conflicting fields from r_state if needed
                    else:
                        r_result = await r_task
                        # merge retrieved chunks and FAQ fields from r_state
                        state["retrieved_chunks"] = r_result.get("retrieved_chunks", [])
                        state["direct_faq_answer"] = r_result.get("direct_faq_answer")
                        state["faq_matched"] = r_result.get("faq_matched", False)
                        state["faq_verified"] = r_result.get("faq_verified", False)
                        state["greeting_only"] = r_result.get("greeting_only")
                        state["rewritten_query"] = r_result.get("rewritten_query", "")
                        # merge stage_timing
                        if "stage_timing_ms" in r_result:
                            if "stage_timing_ms" not in state or state["stage_timing_ms"] is None:
                                state["stage_timing_ms"] = {}
                            state["stage_timing_ms"].update(r_result.get("stage_timing_ms", {}))
                    if state.get("blocked"):
                        content = state.get("refusal_message") or "متأسفم، نمی‌توانم پاسخ دهم."
                        yield ": agent validate_input done blocked=true\n\n"
                        for tok in re.findall(r"\S+\s*", content):
                            yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{'content':tok}}]}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':'content_filter'}]}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    retrieved = state.get("retrieved_chunks", [])
                    yield f": agent retrieve done chunks={len(retrieved)} latency_ms={int((time.monotonic()-t_retrieve)*1000)}\n\n"

                    # Greeting shortcut — no LLM, stream directly
                    if state.get("greeting_only"):
                        state = await node_format(state)
                        content = state.get("formatted_response", {}).get("content", "")
                        yield ": agent format_response greeting\n\n"
                        for tok in re.findall(r"\S+\s*", content):
                            yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{'content':tok}}]}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':'stop'}]}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    # Build context
                    yield ": agent build_context running\n\n"
                    state = await node_build(state)
                    prompt_messages = state.get("prompt_messages", [])
                    yield f": agent build_context done prompt_tokens={len(str(prompt_messages))}\n\n"

                    # FAQ direct — stream canned answer directly (no LLM latency) OR via LLM rewrite
                    # Only use verified FAQ answers; unverified ones fall through to retrieval
                    if state.get("faq_verified") and state.get("direct_faq_answer"):
                        # Option A: direct canned (instant) — uncomment to use LLM rewrite instead
                        content = state.get("direct_faq_answer")
                        # For LLM rewrite, comment above and use Guardrails streaming below with prompt_messages
                        yield ": agent guarded_generate faq_direct streaming\n\n"
                        # Stream without artificial sleep — just chunk by words for UI, no delay
                        for tok in re.findall(r"\S+\s*", content):
                            yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{'content':tok}}]}, ensure_ascii=False)}\n\n"
                            # No sleep — real streaming speed is LLM speed; for canned we can yield immediately
                        yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'rag':{'request_id':request_id,'citations':[]}}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        try:
                            from .tracing import update_trace
                            update_trace(request_id, content[:2000], {"finish_reason":"stop","faq":True,"stream":True})
                        except Exception:
                            pass
                        return

                    # Normal RAG — REAL streaming from Guardrails->Gemma (no sleep, tokens arrive as LLM generates)
                    yield ": agent guarded_generate streaming\n\n"
                    settings = get_settings()
                    req = GuardrailsChatRequest(
                        model=settings.upstream_llm_model,
                        messages=prompt_messages,
                        max_tokens=1024,
                        temperature=0.2,
                    )
                    full_answer = []
                    async with GuardrailsClient() as gclient:
                        async for token in gclient.chat_completion_stream(req, request_id):
                            full_answer.append(token)
                            chunk = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                            }
                            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            # No sleep — real LLM pacing (~30ms/token) is the animation

                    # Finalize state for citations/audit
                    answer_text = "".join(full_answer)
                    state["answer"] = answer_text
                    # Format citations (same as format_response)
                    state = await node_format(state)
                    formatted = state.get("formatted_response", {}) or {}
                    citations = formatted.get("citations", [])
                    finish_reason = formatted.get("finish_reason", "stop")
                    final_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        "rag": {"request_id": request_id, "citations": citations},
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    try:
                        from .tracing import update_trace
                        update_trace(request_id, answer_text[:2000], {"finish_reason":finish_reason,"citations":len(citations),"retrieved":len(retrieved),"stream":True,"real_stream":True})
                    except Exception:
                        pass
                except Exception as e:
                    log.exception("Streaming failed: %s", e)
                    err = {"id":f"chatcmpl-{request_id}","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"خطا در پردازش درخواست."},"finish_reason":"error"}]}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream", headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            })

        # --- Non-streaming (buffered) branch ---
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
        # Compute latency first so it can be included in enriched audit
        latency_ms = int((time.monotonic() - t0) * 1000)
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
            # Enriched audit fields
            guardrail_decision = final_state.get("guardrail_decision")
            if isinstance(guardrail_decision, dict):
                guardrail_input_signals = guardrail_decision.get("signals", []) or []
            else:
                guardrail_input_signals = []
            # Detect retrieval method from chunk sources
            sources = {str(c.get("source", "")).lower() for c in retrieved}
            if "both" in sources or ({"rrf", "ce"} & sources):
                retrieval_method = "rrf+ce"
            elif "rrf" in sources:
                retrieval_method = "rrf"
            elif "ce" in sources:
                retrieval_method = "ce"
            elif retrieved:
                retrieval_method = "rrf+ce"
            else:
                retrieval_method = ""
            # Build audit
            audit = AuditTrail(
                request_id=request_id,
                query=final_state.get("query", ""),
                original_query=final_state.get("query", ""),
                rewritten_query=final_state.get("rewritten_query", ""),
                guardrail_input=guardrail_decision,
                guardrail_input_signals=guardrail_input_signals,
                guardrail_output_signals=[],
                retrieved_chunks=[
                    {
                        "chunk_id": c.get("chunk_id", ""),
                        "title": c.get("title", ""),
                        "heading": c.get("heading", ""),
                        "content": c.get("content", "")[:500],
                        "score": c.get("score", 0),
                        "source": c.get("source", ""),
                        "rank_rrf": c.get("rank_rrf"),
                        "rank_ce": c.get("rank_ce"),
                        "hybrid_score": c.get("hybrid_score"),
                        "rerank_score": c.get("rerank_score"),
                    }
                    for c in retrieved[:10]
                ],
                reranker_scores=[c.get("score", 0) for c in retrieved[:10]],
                context_sent_to_gemma=context_str,
                context_preview=context_str[:500],
                prompt_messages_preview=[{"role": m.get("role", ""), "content": m.get("content", "")[:500]} for m in prompt_msgs[:2]],
                raw_model_output=raw_output[:2000],
                guardrail_output={"blocked": final_state.get("blocked", False), "refusal": final_state.get("refusal_message")},
                final_answer=content,
                citations=[Citation(**c) for c in citations],
                model=request.model,
                stage_timing_ms=final_state.get("stage_timing_ms", {}),
                retrieval_method=retrieval_method,
                retrieval_chunks_total=len(retrieved),
                retrieval_chunks_used=len(citations),
                reranker_applied=len(retrieved) > 0,
                latency_ms=latency_ms,
            )
        except Exception as e:
            log.warning(f"Failed to build audit trail: {e}")
            audit = None

        # Build response
        # latency_ms already computed above
        # Populate audit latency if not already set
        if audit is not None and audit.latency_ms is None:
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

        # Auto-save to observability dashboard (fire-and-forget, non-blocking)
        # Skip internal task requests (follow-up generation etc.): they have audit.request_id == query == "### Task:..."
        try:
            import os as _os
            _is_task = False
            try:
                _q = (final_state.get("query") or "").strip()
                if _q.startswith("### Task:"):
                    _is_task = True
            except Exception:
                pass
            obs_url = _os.getenv("OBSERVABILITY_URL", "http://127.0.0.1:3000")
            if obs_url and audit is not None and not _is_task:
                _audit_dict = audit.model_dump()
                _payload = {
                    "request_id": request_id,
                    "user_query": _audit_dict.get("query", ""),
                    "rewritten_query": _audit_dict.get("rewritten_query", ""),
                    "response": _audit_dict.get("final_answer", ""),
                    "stage_timing": _audit_dict.get("stage_timing_ms", {}),
                    "guardrail_input": _audit_dict.get("guardrail_input"),
                    "guardrail_output": _audit_dict.get("guardrail_output"),
                    "retrieved_chunks": _audit_dict.get("retrieved_chunks", []),
                    "citations": [dict(c) for c in citations] if citations else [],
                    "context_preview": _audit_dict.get("context_preview", ""),
                    "model": _audit_dict.get("model", ""),
                    "latency_ms": latency_ms,
                }
                async def _post_observability_async(payload=_payload, obs=obs_url):
                    for base in [obs, "http://rag-tracing-fallback:3000"]:
                        try:
                            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as _cl:
                                await _cl.post(f"{base.rstrip('/')}/api/observability/traces", json=payload)
                            break
                        except Exception:
                            continue
                asyncio.create_task(_post_observability_async())
        except Exception:
            pass

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

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

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




if __name__ == "__main__":
    main()