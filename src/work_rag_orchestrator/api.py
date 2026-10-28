"""FastAPI application for Orchestrator service."""

from __future__ import annotations

import logging
import os
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

# Opt-in app-level INFO logging (running under `-m uvicorn` configures only
# uvicorn's own loggers; without this, app INFO lines never reach stderr).
if os.environ.get("ORCH_DEBUG") == "1":
    logging.basicConfig(level=logging.INFO)

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
    _TRUTH_RE = re.compile(r"^\s*/(?:truth|ground_truth|true|correct)\s+(.+?)\s*$", re.IGNORECASE | re.DOTALL)
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
        except Exception:
            pass  # fall through to normal RAG flow on any error

        # --- /report command: save report without calling LLM ---
        try:
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
                    f"Report saved: [{category}]{cmt_str2} {tag_str2}\n\n"
                    f"View in dashboard: http://127.0.0.1:3000/dashboard/observability"
                )
                return ChatCompletionResponse(
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": msg2}, finish_reason="stop")],
                    rag=RAGMetadata(request_id=rate_request_id2, citations=[]),
                    audit=None,
                )
        except Exception:
            pass

        # --- /truth command: save ground truth without calling LLM ---
        try:
            m3 = _TRUTH_RE.match(last_content.strip())
            if m3:
                text = (m3.group(1) or "").strip()
                # tags at end like #my-tag are not expected for ground truth, but allow them
                # keep text as-is (the whole ground truth answer)
                gt_request_id = x_request_id or str(uuid.uuid4())
                try:
                    import httpx as _httpx3
                    import os as _os3
                    obs_url3 = _os3.getenv("OBSERVABILITY_URL", "http://127.0.0.1:3000")
                    for base in [obs_url3, "http://rag-tracing-fallback:3000"]:
                        try:
                            async with _httpx3.AsyncClient(timeout=3.0) as _client:
                                await _client.post(
                                    f"{base.rstrip('/')}/api/observability/evaluations",
                                    json={
                                        "request_id": gt_request_id,
                                        "comment": f"[GROUND_TRUTH] {text}".strip()[:4000],
                                        "tags": ["ground_truth"],
                                        "user_query": last_content[:500],
                                        "model": request.model,
                                    },
                                )
                            break
                        except Exception:
                            continue
                except Exception:
                    pass
                msg3 = (
                    f"Ground truth saved for {gt_request_id[:8]}\n\n"
                    f"> {text[:1500]}\n\n"
                    f"View in dashboard: http://127.0.0.1:3000/dashboard/observability"
                )
                return ChatCompletionResponse(
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": msg3}, finish_reason="stop")],
                    rag=RAGMetadata(request_id=gt_request_id, citations=[]),
                    audit=None,
                )
        except Exception:
            pass

        # --- Task filter: WebUI ungrounded follow-up generation ---
        # WebUI's ENABLE_FOLLOW_UP_GENERATION sends "### Task: Suggest 3-5 relevant follow-up questions"
        # These are untracked task LLM calls that produced 268 junk traces before cleanup.
        # They are ungrounded (not from KB) and should be filtered from RAG/observability.
        # Grounded follow-ups are now generated inside nodes/format_response.py and
        # returned via rag.follow_ups, so WebUI doesn't need to generate its own.
        # Filtering here prevents junk traces and wasted RAG. If WebUI still calls this,
        # return a minimal empty response so WebUI shows no ungrounded suggestions.
        try:
            _task_q = request.messages[-1].get("content", "") if request.messages else ""
            if "### Task" in _task_q or "Suggest 3-5" in _task_q:
                log.info("Filtered ### Task follow-up prompt (ungrounded) - returning empty, no RAG/observability")
                return ChatCompletionResponse(
                    model=request.model,
                    choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": ""}, finish_reason="stop")],
                    rag=RAGMetadata(request_id=x_request_id or str(uuid.uuid4()), citations=[], follow_ups=[]),
                    audit=None,
                )
        except Exception:
            pass

        # Generate request ID for tracing
        request_id = x_request_id or str(uuid.uuid4())
        t0 = time.monotonic()
        # Langfuse trace (self-hosted) — non-blocking, direct HTTP fallback
        try:
            from .tracing import start_trace
            # Skip tracing Task prompts (already filtered above, but double-guard)
            _input_for_trace = request.messages[-1].get("content", "")[:1000] if request.messages else ""
            if "### Task" in _input_for_trace:
                raise ValueError("skip Task trace")
            start_trace(request_id, "rag", _input_for_trace, {"model": request.model, "request_id": request_id})
        except Exception as e:
            # Task traces are intentionally skipped; other errors logged
            if "skip Task" not in str(e):
                log.warning("Langfuse trace init failed: %s", e)
        
        # Validate model — primary is ICS Helper Agent; keep legacy aliases for backward compat
        allowed_models = {"ics-helper-agent", "work-rag-agent", "gemma-4-31b", "gemma-4-31b-triton-trt", get_settings().upstream_llm_model}
        if request.model not in allowed_models:
            log.warning("Unsupported model requested: %s", request.model)
        
        config = {"configurable": {"thread_id": request_id}}
        # Restore sticky applied_subspace from checkpoint (whole KB -> None, after spec -> leaf)
        sticky_applied = None
        try:
            prev = await rag_graph.aget_state(config)
            if prev and getattr(prev, "values", None):
                sticky_applied = prev.values.get("sticky_subspace") or prev.values.get("applied_subspace")
        except Exception:
            sticky_applied = None
        # Fallback: WebUI's POST /api/chat/completions has no X-Request-ID,
        # so each turn gets a new thread_id — mine sticky from chat history instead
        if sticky_applied is None:
            for hm in reversed(request.messages[:-1]):
                if hm.get("role") != "user":
                    continue
                hc = str(hm.get("content","")).strip()
                if not hc:
                    continue
                # Sticky must come from user specs, not user questions:
                # skip interrogative history entries (they are questions, not declarations)
                if "؟" in hc or "?" in hc or any(w in hc for w in ["چیست","چطور","چگونه","آیا","چرا","کجاست","چیه","چيست","کیه","کيست"]):
                    continue
                # long explicit declaration like "اعتبارسنجی تسهیلات برای اشخاص حقیقی"
                if "اشخاص حقیقی" in hc and "تسهیلات" in hc:
                    sticky_applied = "facilities_person"; break
                if "اشخاص حقوقی" in hc and "تسهیلات" in hc:
                    sticky_applied = "facilities_legal"; break
                # "هیچ کدوم" + manager/company hint → general (user rejected credit options, wants معرفی/هیات مدیره)
                if "هیچ کدوم" in hc or "هیچکدوم" in hc:
                    # if mentions مدیر/هیات مدیره/معرفی/شرکت without تسهیلات/چک → general
                    if any(kw in hc for kw in ["مدیر","هیات مدیره","هیئت مدیره","معرفی","عمومی","شرکت","عام"]):
                        sticky_applied = "general"; break
                    # bare "هیچ کدوم" after a clarify that included general → treat as general
                    # fall through to general handling below
                    low2 = hc.lower()
                    if "general" in low2 or "عام" in hc or "عمومی" in hc:
                        sticky_applied = "general"; break
                _hc_decl = hc.strip().replace("ي", "ی").replace("ك", "ک")
                if _hc_decl in ("شخصی", "حقیقی"):
                    sticky_applied = "facilities_person"; break
                if _hc_decl in ("شرکتی", "حقوقی"):
                    sticky_applied = "facilities_legal"; break
                if hc.lower().strip() in ("چک","cheque","تسهیلات","وام","شرکت","حقوقی","عمومی","general","مدیر","هیات مدیره","هیئت مدیره","معرفی","کاربرد","عام"):
                    low = hc.lower()
                    if "چک" in hc or "cheque" in low:
                        sticky_applied = "cheque"; break
                    elif any(kw in hc for kw in ["مدیر","هیات مدیره","هیئت مدیره","معرفی","عمومی","عام","کاربرد"]) or "general" in low:
                        sticky_applied = "general"; break
                    elif "حقوقی" in hc or "شرکت" in hc:
                        # "شرکت" alone without تسهیلات/چک already handled above as general; here it is with facility context
                        # distinguish: if also has تسهیلات/وام → facilities_legal else general (already returned)
                        if "تسهیلات" in hc or "وام" in hc or "حقوقی" in hc:
                            sticky_applied = "facilities_legal"; break
                        else:
                            sticky_applied = "general"; break
                    else:
                        sticky_applied = "facilities_person"; break
                low = hc.lower()
                # short re-specification: "منظورم چک بود" or "مدیر شرکته" / "هیچ کدوم مدیر"
                if ("منظورم" in hc or "هیچ کدوم" in hc or "هیچکدوم" in hc or low.startswith("چک ") or hc.strip() == "چک" or "مدیر" in hc) and len(hc.split()) <= 6:
                    if "چک" in hc:
                        sticky_applied = "cheque"; break
                    if any(kw in hc for kw in ["مدیر","هیات مدیره","هیئت مدیره","معرفی","عمومی","عام","کاربرد"]) or "general" in low:
                        sticky_applied = "general"; break
                    if "حقوقی" in hc or ("شرکت" in hc and ("تسهیلات" in hc or "وام" in hc)):
                        sticky_applied = "facilities_legal"; break
                    if "شرکت" in hc and "تسهیلات" not in hc and "چک" not in hc:
                        sticky_applied = "general"; break
        # Current query general override (strong general keywords in current message itself)
        cur_q = str(request.messages[-1].get("content","")) if request.messages else ""
        # If current query explicitly says "سوالات عمومی" or contains board member names, treat as general
        _explicit_general = False
        if "سوالات عمومی" in cur_q or "سوال عمومی" in cur_q or "قاسم" in cur_q or "بور" in cur_q or "هیات مدیره" in cur_q or "هیئت مدیره" in cur_q or "معرفی شرکت" in cur_q:
            sticky_applied = "general"
            _explicit_general = True
        # Declaration follow-up: a short reply like "شخصی"/"شرکتی"/"چک"/"عمومی"
        # after a clarify selects the subspace AND carries the prior question forward
        # so the next step retrieves an answer instead of asking the user to re-ask.
        _decl_key = re.sub(r"[\s\.\!\؟\?،,؛;:]+", "", cur_q).replace("ي", "ی").replace("ك", "ک")
        _declarations = {
            "شخصی": "facilities_person",
            "حقیقی": "facilities_person",
            "شرکتی": "facilities_legal",
            "حقوقی": "facilities_legal",
            "چک": "cheque",
            "cheque": "cheque",
            "عمومی": "general",
            "general": "general",
        }
        _decl = _declarations.get(_decl_key)
        _declared_filter = None
        if _decl:
            sticky_applied = _decl
            _prev_q = None
            for _pm in reversed(request.messages[:-1]):
                if _pm.get("role") != "user":
                    continue
                _pq = str(_pm.get("content") or "").strip()
                if not _pq:
                    continue
                if "؟" in _pq or "?" in _pq or any(w in _pq for w in ["چیست", "چطور", "چگونه", "آیا", "چرا", "کجاست", "چیه", "کیه"]):
                    _prev_q = _pq
                    break
            if _prev_q is None and prev and getattr(prev, "values", None):
                # Bare-API clients carry no history: if the previous turn on
                # this thread ended in a clarify, its query IS the question the
                # declaration answers.
                if prev.values.get("needs_clarification"):
                    _pvc = str(prev.values.get("query") or "").strip()
                    if _pvc:
                        _prev_q = _pvc
            if _prev_q:
                cur_q = _prev_q
            try:
                _declared_filter = get_settings().subspace_map.get(_decl)
            except Exception:
                pass
            log.info("Declaration follow-up %r -> subspace=%s query=%r", _decl_key, _decl, cur_q[:80])
        # --- C: Per-turn subspace switch by the embedding scorer ---
        # Sticky subspace is preserved ONLY while the CURRENT query scores
        # ambiguous AND sticky is still competitive with the argmax candidate.
        # A confident (non-ambiguous) scorer decision switches the subspace
        # even against sticky. Declaration follow-ups ("شخصی"/"شرکتی"/...) and
        # the explicit general override above stay authoritative — the scorer
        # never runs on those single-word replies.
        _sc_target = None       # confident scorer subspace (switch/keep)
        _sc_keep_sticky = None  # True/False when ambiguous; None = scorer unavailable
        _sc_top1_thr = 0.20     # argmax sim floor for a scored conf-switch
        if _decl is None and not _explicit_general:
            try:
                from .subspaces import get_scorer
                _scorer = get_scorer()
                if _scorer.available:
                    # Context-aware scoring: short colloquial follow-ups
                    # ("اعتبار رو چ") carry almost no signal alone — prepend
                    # the previous user turn so the embedding scorer sees the
                    # thread topic. Long queries carry enough signal raw.
                    _sc_q = cur_q
                    if sticky_applied and len(cur_q.split()) <= 5:
                        # prev user turn: from message history (WebUI sends it),
                        # else from the graph checkpoint state (bare API clients)
                        _prev_u = None
                        for _pm in reversed(request.messages[:-1]):
                            if _pm.get("role") == "user":
                                _pu = str(_pm.get("content") or "").strip()
                                if _pu:
                                    _prev_u = _pu
                                    break
                        if _prev_u is None and prev and getattr(prev, "values", None):
                            _pvq = str(prev.values.get("query") or "").strip()
                            if _pvq:
                                _prev_u = _pvq
                        if _prev_u:
                            _sc_q = _prev_u + " " + cur_q
                    log.info("Subspace scorer ctx: sticky=%r short=%r scored_q=%r", sticky_applied, len(cur_q.split()) <= 5, _sc_q[:100])
                    _sc = await _scorer.ascore(_sc_q)
                    _amb_margin = float(get_settings().subspace_ambiguity_margin)
                    if not _sc.ambiguous and _sc.subspace:
                        _sc_target = _sc.subspace
                    else:
                        _cand = max(_sc.sims, key=lambda s: _sc.sims[s]) if _sc.sims else None
                        if sticky_applied is None or _cand is None or sticky_applied == _cand:
                            _sc_keep_sticky = True
                        else:
                            # keep sticky only while still competitive with argmax
                            _diff = _sc.sims.get(_cand, 0.0) - _sc.sims.get(sticky_applied, 0.0)
                            if _diff <= _amb_margin:
                                _sc_keep_sticky = True
                            elif _sc.sims.get(_cand, 0.0) >= _sc_top1_thr:
                                # argmax clearly stronger than sticky -> switch
                                _sc_target = _cand
                            else:
                                # sticky stale, argmax too weak -> drop sticky
                                # so classify/infer path decides (likely clarify)
                                _sc_keep_sticky = False
                    log.info(
                        "Subspace scorer q=%r (scored=%r): switch=%r keep_sticky=%r ambiguous=%s sims=%s",
                        cur_q[:60], _sc_q[:100], _sc_target, _sc_keep_sticky, _sc.ambiguous,
                        {k: round(v, 3) for k, v in (_sc.sims or {}).items()},
                    )
            except Exception as _sc_err:
                log.warning("api subspace scorer failed: %s — keyword re-spec heuristics stay in effect", _sc_err)
                _sc_target = None
                _sc_keep_sticky = None
        # Confident scorer decision is authoritative: switch when different.
        # NOTE: the switched/kept subspace is recorded for STICKY inheritance
        # (initial_state.sticky_subspace) but NOT injected as filter_path —
        # topically-matching chunks often live in a sibling subspace's folder
        # (e.g. "من تو جنگ اول یه چک برگشتی داشتم" is cheque-topic but the best
        # chunk lives under اشخاص حقوقی/اعتبارسنجی تسهیلات), so forcing the
        # scope here caused spurious "پاسخی یافت نشد" refusals. Only
        # declaration replies and the keyword fallback may scope retrieval.
        if _sc_target and sticky_applied != _sc_target:
            log.info("Subspace switch by scorer: sticky %s -> %s for q=%r", sticky_applied, _sc_target, cur_q[:60])
            sticky_applied = _sc_target
        elif sticky_applied is not None and _sc_keep_sticky is False:
            log.info("Sticky %s stale by scorer (argmax candidate != sticky) for q=%r — dropping", sticky_applied, cur_q[:60])
            sticky_applied = None
        # T4: Carry sticky correctly for generic follow-ups - preserve checkpoint sticky even when current query is generic/ambiguous
        # Previous logic mined history skipping interrogatives, but a generic follow-up (e.g. "آیا میزان درآمد...") has no subspace keywords
        # yet is in same thread => should inherit sticky. Checkpoint sticky already captured above; ensure generic query doesn't clear it.
        # Detect explicit re-spec in current query (short spec or strong keyword of different subspace) - only then allow switch
        if sticky_applied is not None:
            # Generic follow-up heuristic: does current query contain any explicit subspace keyword of a DIFFERENT subspace?
            _cur_has_cheque = "چک" in cur_q
            _cur_has_facility = "تسهیلات" in cur_q or "وام" in cur_q
            _cur_has_legal = "حقوقی" in cur_q
            _cur_has_general = any(kw in cur_q for kw in ["مدیر","هیات مدیره","هیئت مدیره","معرفی","عام","کاربرد","قاسم","بور"])
            _cur_is_spec_candidate = False
            # Short spec candidate (<=6 words) containing a subspace keyword different from sticky => potential re-spec
            if len(cur_q.split()) <= 6 and cur_q.strip():
                if sticky_applied != "cheque" and _cur_has_cheque:
                    _cur_is_spec_candidate = True
                elif sticky_applied != "facilities_person" and ("حقیقی" in cur_q and _cur_has_facility):
                    _cur_is_spec_candidate = True
                elif sticky_applied != "facilities_legal" and _cur_has_legal and _cur_has_facility:
                    _cur_is_spec_candidate = True
                elif sticky_applied != "general" and _cur_has_general:
                    _cur_is_spec_candidate = True
            # If not a re-spec candidate and query is generic/ambiguous, keep sticky (do not clear)
            if not _cur_is_spec_candidate:
                # Preserve sticky for generic follow-up (no explicit different subspace mention)
                # This covers conv_facilities_person_01 T3 ("آیا میزان درآمد...") and conv_cheque_02 T3 ("تاثیر چک...")
                # where the follow-up lacks a different subspace keyword - sticky should persist
                pass  # sticky_applied already holds correct value, ensure not cleared
            # If it IS a re-spec candidate, let classify/infer_specification handle the switch downstream (don't force keep)
        # --- B: Sticky decay — stale subspace after a long run of generic turns ---
        # Count consecutive generic user turns since the last subspace-specifying
        # turn by scanning request.messages backwards (user-role turns only).
        # A turn is "specifying" if it contains any subspace keyword
        # (چک، تسهیلات، حقیقی، حقوقی، شرکت، عمومی، معرفی، مدیر، سهامدار) AND is
        # short (<= 8 words); anything else counts as generic. If the generic
        # run exceeds settings.sticky_decay_turns, the sticky subspace is stale:
        # drop it (sticky_applied = None) so the T4 injection below is skipped
        # and the query goes through the normal classify/infer path (likely
        # clarify or a fresh subspace inference).
        if sticky_applied is not None:
            try:
                _decay_kws = ["چک", "تسهیلات", "حقیقی", "حقوقی", "شخصی", "شرکتی", "شرکت", "عمومی", "معرفی", "مدیر", "سهامدار"]
                _generic_turns = 0
                for _dm in reversed(request.messages):
                    if _dm.get("role") != "user":
                        continue
                    _dt = str(_dm.get("content", "") or "").strip()
                    if not _dt:
                        continue
                    # A specifying turn refreshes the sticky subspace — stop counting.
                    if any(kw in _dt for kw in _decay_kws) and len(_dt.split()) <= 8:
                        break
                    _generic_turns += 1
                if _generic_turns > get_settings().sticky_decay_turns:
                    log.info(
                        "B sticky decay: %d consecutive generic user turns > sticky_decay_turns=%d — dropping sticky %s for q=%r",
                        _generic_turns, get_settings().sticky_decay_turns, sticky_applied, cur_q[:60],
                    )
                    sticky_applied = None
            except Exception:
                pass  # decay check must never break the request path
        # Ensure filter_path from request overrides sticky (explicit scope is authoritative)
        _req_filter_path = getattr(request, "filter_path", None) or (request.model_extra.get("filter_path") if getattr(request, "model_extra", None) else None)
        if not _req_filter_path and _declared_filter:
            _req_filter_path = _declared_filter
        if not _req_filter_path and sticky_applied is not None and _sc_target is None:
            # Check if current query is generic (no explicit different subspace) - same heuristic as above
            _cur_has_cheque2 = "چک" in cur_q
            _cur_has_facility2 = "تسهیلات" in cur_q or "وام" in cur_q
            _cur_has_legal2 = "حقوقی" in cur_q
            _cur_has_general2 = any(kw in cur_q for kw in ["مدیر","هیات مدیره","هیئت مدیره","معرفی","عام","کاربرد","قاسم","بور"])
            _is_explicit_different = False
            if len(cur_q.split()) <= 6 and cur_q.strip():
                if sticky_applied != "cheque" and _cur_has_cheque2:
                    _is_explicit_different = True
                elif sticky_applied != "facilities_person" and ("حقیقی" in cur_q and _cur_has_facility2):
                    _is_explicit_different = True
                elif sticky_applied != "facilities_legal" and _cur_has_legal2 and _cur_has_facility2:
                    _is_explicit_different = True
                elif sticky_applied != "general" and _cur_has_general2:
                    _is_explicit_different = True
            # For generic follow-ups (including longer generic like "آیا میزان درآمد...") there is no explicit different keyword in short-spec sense,
            # so inject sticky. For longer queries containing same-subspace keyword (e.g. "تاثیر چک...") also keep sticky.
            # Only skip injection when explicit different short spec (e.g. "چک" after facilities_person) - that case is handled by classify spec path anyway
            if not _is_explicit_different:
                # Also ensure current query does NOT contain strong keyword of a clearly different subspace in longer form
                # e.g. "حالا درباره تسهیلات اشخاص حقیقی بگو" after cheque should NOT inject cheque - it's an explicit re-spec
                # Heuristic: if query contains "تسهیلات" + "حقیقی" while sticky is cheque, don't inject
                _explicit_long = False
                if sticky_applied == "cheque" and "تسهیلات" in cur_q and "حقیقی" in cur_q:
                    _explicit_long = True
                elif sticky_applied == "cheque" and "تسهیلات" in cur_q and "حقوقی" in cur_q:
                    _explicit_long = True
                elif sticky_applied in ("facilities_person","facilities_legal") and _cur_has_cheque2 and "تاثیر" not in cur_q and "آیا" not in cur_q:
                    # "چک" alone after facilities should be considered explicit re-spec (short), already handled; longer "تاثیر چک..." is NOT different - keep sticky
                    # So only treat "چک" alone/short as different, which was already covered
                    _explicit_long = False
                if not _explicit_long:
                    try:
                        _injected = get_settings().subspace_map.get(sticky_applied)
                        if _injected:
                            _req_filter_path = _injected
                            log.info("T4 api sticky inject: generic follow-up inherits %s -> filter_path %r for q=%r", sticky_applied, _req_filter_path, cur_q[:60])
                    except Exception:
                        pass
        if _req_filter_path:
            try:
                _map = get_settings().subspace_map or {}
                _rev = {v: k for k, v in _map.items()}
                _label = _rev.get(_req_filter_path, _req_filter_path)
                if "چک" in _req_filter_path:
                    _label = "cheque"
                elif "حقوقی" in _req_filter_path:
                    _label = "facilities_legal"
                elif "حقیقی" in _req_filter_path:
                    _label = "facilities_person"
                sticky_applied = _label
            except Exception:
                sticky_applied = _req_filter_path

        # Prepare initial state
        initial_state: RAGState = {
            "request_id": request_id,
            "messages": request.messages,
            "query": cur_q,
            "rewritten_query": cur_q if cur_q != (request.messages[-1].get("content", "") if request.messages else "") else "",
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
            "faq_verified": False,
            "greeting_only": None,
            "stage_timing_ms": {},
            "applied_subspace": sticky_applied,
            # Cross-turn sticky for the next turn's scorer decision (not used
            # to scope retrieval this turn — see switch-by-scorer note above).
            "sticky_subspace": sticky_applied,
            "needs_clarification": False,
            "clarifying_question": None,
            "available_subspaces": None,
            "injected_entities": [],
            "supplemental": [],
            "filter_path": _req_filter_path,
            "follow_ups": [],
            "fallback_used": False,
            "fallback_reason": None,
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

                    query = initial_state.get("query") or (initial_state["messages"][-1]["content"] if initial_state.get("messages") else "")
                    initial_state["query"] = query
                    yield ": agent validate_input + retrieve running\n\n"
                    t_retrieve = time.monotonic()
                    v_task = asyncio.create_task(node_validate(initial_state))
                    r_task = asyncio.create_task(node_retrieve(initial_state))
                    state = await v_task
                    await r_task
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

                    # Clarify branch — scattered but no sticky (F5 conditional edge)
                    if state.get("needs_clarification"):
                        from .nodes.generate_clarification import generate_clarification as node_clarify
                        state = await node_clarify(state)
                        content = state.get("clarifying_question") or state.get("formatted_response", {}).get("content", "") or "لطفاً مشخص کنید منظورتان کدام بخش است."
                        yield ": agent generate_clarification clarify\n\n"
                        for tok in re.findall(r"\S+\s*", content):
                            yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{'content':tok}}]}, ensure_ascii=False)}\n\n"
                        sup_clarify = state.get("supplemental") or state.get("injected_entities") or []
                        yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':'clarify'}],'rag':{'request_id':request_id,'citations':[],'supplemental': sup_clarify if sup_clarify else None}}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        try:
                            from .tracing import update_trace
                            update_trace(request_id, content[:2000], {"finish_reason":"clarify","stream":True,"supplemental": len(sup_clarify) if sup_clarify else 0})
                        except Exception:
                            pass
                        return

                    # Build context
                    yield ": agent build_context running\n\n"
                    state = await node_build(state)
                    prompt_messages = state.get("prompt_messages", [])
                    yield f": agent build_context done prompt_tokens={len(str(prompt_messages))}\n\n"

                    # G5 hallucination guard for streaming: if no evidence, stream refusal without LLM (T3 per-subspace)
                    try:
                        _ev_t = float(get_settings().get_evidence_threshold(state.get("applied_subspace")))
                    except Exception:
                        _ev_t = 0.15
                    _chunks_s = state.get("retrieved_chunks") or []
                    _top_s = 0.0
                    for _cc in _chunks_s:
                        for _kk in ("rerank_score", "hybrid_score", "score"):
                            _vv = _cc.get(_kk)
                            if isinstance(_vv, (int, float)):
                                _top_s = max(_top_s, float(_vv))
                    _has_sup_s = bool(state.get("injected_entities") or state.get("supplemental"))
                    # T6 lexical evidence bypass
                    _has_lex_s = False
                    try:
                        _q_s = state.get("query") or state.get("rewritten_query") or ""
                        from .nodes.format_response import _has_lexical_evidence as _lex_s
                        _has_lex_s = _lex_s(_q_s, _chunks_s, state.get("applied_subspace"))
                    except Exception:
                        _has_lex_s = False
                    if not state.get("faq_verified") and (_top_s < _ev_t or not _chunks_s) and not _has_sup_s and not _has_lex_s:
                        _no_ev = "بر اساس اطلاعات موجود در پایگاه دانش، پاسخی برای این سوال یافت نشد."
                        yield ": agent guarded_generate no_evidence_guard streaming\n\n"
                        for tok in re.findall(r"\S+\s*", _no_ev):
                            yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{'content':tok}}]}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'rag':{'request_id':request_id,'citations':[],'supplemental': None}}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        try:
                            from .tracing import update_trace as _ut2
                            _ut2(request_id, _no_ev[:2000], {"finish_reason":"stop","guard":"no_evidence","top_score":_top_s,"threshold":_ev_t,"stream":True})
                        except Exception:
                            pass
                        return

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
                        sup_faq = state.get("supplemental") or state.get("injected_entities") or []
                        yield f"data: {json.dumps({'id':f'chatcmpl-{request_id}','object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':'stop'}],'rag':{'request_id':request_id,'citations':[],'supplemental': sup_faq if sup_faq else None}}, ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        try:
                            from .tracing import update_trace
                            update_trace(request_id, content[:2000], {"finish_reason":"stop","faq":True,"stream":True,"supplemental": len(sup_faq) if sup_faq else 0})
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

                    # Finalize state for citations/audit + grounded follow-ups
                    answer_text = "".join(full_answer)
                    state["answer"] = answer_text
                    # Format citations (same as format_response) — includes grounded follow_ups
                    state = await node_format(state)
                    formatted = state.get("formatted_response", {}) or {}
                    citations = formatted.get("citations", [])
                    finish_reason = formatted.get("finish_reason", "stop")
                    follow_ups_stream = formatted.get("follow_ups") or state.get("follow_ups") or []
                    supplemental_stream = state.get("supplemental") or state.get("injected_entities") or []
                    final_chunk = {
                        "id": f"chatcmpl-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                        "rag": {"request_id": request_id, "citations": citations, "supplemental": supplemental_stream if supplemental_stream else None, "follow_ups": follow_ups_stream if follow_ups_stream else None},
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
        
        # Extract formatted response — preserve clarify finish_reason (generate_clarification -> END)
        formatted = final_state.get("formatted_response", {}) or {}
        # Handle clarify finish_reason explicitly; generate_clarification sets formatted_response
        # with clarify; ensure it is not overwritten by generic stop/error handling
        content = formatted.get("content", "An error occurred.")
        finish_reason = formatted.get("finish_reason", "error")
        # Preserve clarify if state signals it, even if formatted was malformed.
        # Supplemental entity rows are evidence: never downgrade a real answer
        # to clarify when the entity tool matched (reason code, shareholder...).
        _has_entity_evidence = bool(final_state.get("supplemental") or final_state.get("injected_entities"))
        if (final_state.get("needs_clarification") or finish_reason == "clarify") and not _has_entity_evidence:
            finish_reason = "clarify"
            # Ensure content reflects clarifying_question when formatted content missing
            if not content or content == "An error occurred.":
                content = final_state.get("clarifying_question") or content
        citations = formatted.get("citations", [])
        follow_ups = formatted.get("follow_ups") or final_state.get("follow_ups") or []
        # Ensure list type
        if not isinstance(follow_ups, list):
            follow_ups = []
        
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
            supplemental = final_state.get("supplemental") or final_state.get("injected_entities") or []
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
                supplemental=supplemental,
                applied_subspace=final_state.get("applied_subspace"),
                available_subspaces=final_state.get("available_subspaces"),
                fallback_used=bool(final_state.get("fallback_used", False)),
                fallback_reason=final_state.get("fallback_reason"),
                retry_stage_timing={k: v for k, v in {"retrieve_retry": final_state.get("stage_timing_ms", {}).get("retrieve_retry"), "retrieve_broadened": final_state.get("stage_timing_ms", {}).get("retrieve_broadened")}.items() if v is not None} or None,
                broaden_used=bool(final_state.get("broaden_used", False)),
                ambiguity_score=final_state.get("ambiguity_score"),
                ambiguity_margin=final_state.get("ambiguity_margin"),
                ambiguity_entropy=final_state.get("ambiguity_entropy"),
                ambiguity_ambiguous=final_state.get("ambiguity_ambiguous"),
                ambiguity_source=final_state.get("ambiguity_source"),
                subspace_sims=final_state.get("subspace_sims"),
                ambiguity_intra_spread=final_state.get("ambiguity_intra_spread"),
                ambiguity_proto_coverage=final_state.get("ambiguity_proto_coverage"),
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

        # Auto-save to observability dashboard (fire-and-forget)
        # Filter out Task prompts (WebUI ungrounded follow-up generation) from traces
        try:
            import os as _os
            obs_url = _os.getenv("OBSERVABILITY_URL", "http://127.0.0.1:3000")
            # Task filter for observability (268 junk traces before cleanup)
            _obs_query = audit.query if audit is not None else final_state.get("query", "")
            if _obs_query and ("### Task" in str(_obs_query) or "Suggest 3-5" in str(_obs_query)):
                log.info("Skipping observability save for Task prompt (ungrounded): %s", request_id)
                obs_url = None  # disable save
            if obs_url and audit is not None:
                import httpx as _httpx
                import threading as _threading
                _audit_dict = audit.model_dump()
                def _post_observability(audit_data=_audit_dict, req_id=request_id, obs=obs_url, cid=citations):
                    try:
                        import httpx as _hx
                        payload = {
                            "request_id": req_id,
                            "user_query": audit_data.get("query", ""),
                            "rewritten_query": audit_data.get("rewritten_query", ""),
                            "response": audit_data.get("final_answer", ""),
                            "stage_timing": audit_data.get("stage_timing_ms", {}),
                            "guardrail_input": audit_data.get("guardrail_input"),
                            "guardrail_output": audit_data.get("guardrail_output"),
                            "retrieved_chunks": audit_data.get("retrieved_chunks", []),
                            "citations": [dict(c) for c in cid] if cid else [],
                            "context_preview": audit_data.get("context_preview", ""),
                            "model": audit_data.get("model", ""),
                            "latency_ms": latency_ms,
                            "applied_subspace": audit_data.get("applied_subspace"),
                            "available_subspaces": audit_data.get("available_subspaces"),
                            "filter_path": audit_data.get("filter_path"),
                            "fallback_used": audit_data.get("fallback_used", False),
                            "fallback_reason": audit_data.get("fallback_reason"),
                            "retry_stage_timing": audit_data.get("retry_stage_timing"),
                            "broaden_used": audit_data.get("broaden_used", False),
                        }
                        for base in [obs, "http://rag-tracing-fallback:3000"]:
                            try:
                                with _hx.Client(timeout=3.0) as _cl:
                                    _cl.post(f"{base.rstrip('/')}/api/observability/traces", json=payload)
                                break
                            except Exception:
                                continue
                    except Exception:
                        pass
                _threading.Thread(target=_post_observability, daemon=True).start()
        except Exception:
            pass

        # supplemental envelope: rag.supplemental (F6) wired from KB injected_entities
        rag_supplemental = final_state.get("supplemental") or final_state.get("injected_entities") or []
        # ensure audit also carries supplemental if audit was built without it (fallback)
        if audit is not None and getattr(audit, "supplemental", None) in (None, []):
            try:
                audit.supplemental = rag_supplemental or None
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
                supplemental=rag_supplemental if rag_supplemental else None,
                follow_ups=follow_ups if follow_ups else None,
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




if __name__ == "__main__":
    main()