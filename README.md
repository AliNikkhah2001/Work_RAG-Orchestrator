# Work RAG Orchestrator

LangGraph orchestration and public OpenAI-compatible chat API for the Work Credit RAG platform.

> **Status:** Implemented and live on Vast.ai (`main`). The runtime matches the deployed system and includes the **ambiguity resolver**, **clarifying-question generator**, and **metadata (subspace) filtering** layers.

## 1. Summary

The Orchestrator owns **request coordination**. It receives a chat completion request, runs it through a deterministic LangGraph workflow, and returns a grounded, citation-shaped response.

The graph is deterministic and classifies every turn before retrieval so greetings, subspace specifications, and real questions each take the right path. It never calls Gemma directly — generation is delegated to Guardrails (`POST /v1/chat/completions` → Gemma `:18000`).

## 2. Graph architecture

<details>
<summary>Runtime flow diagram (LangGraph)</summary>

```mermaid
flowchart TD
    Start((Start)) --> VI[validate_input]
    VI -->|blocked| Ref[format_refusal]
    VI -->|allowed| Cl[classify]
    Cl -->|greeting / specification| Fr[format_response]
    Cl -->|question (hybrid on)| Hc[hybrid_classify]
    Cl -->|question (hybrid off)| Ret[retrieve]
    Hc --> Ret[retrieve]
    Ref --> End((End))
    Fr --> End((End))
    Ret -->|needs clarification| Gc[generate_clarification]
    Ret -->|continue| Bc[build_context]
    Gc --> End((End))
    Bc --> Gg[guarded_generate]
    Gg --> Fr
```

Validated against `src/work_rag_orchestrator/graph.py` exactly:
`validate_input → (blocked → format_refusal → End | allowed → classify) → (greeting/specification → format_response → End | question → hybrid_classify → retrieve | question → retrieve when hybrid disabled) → retrieve → (needs clarification → generate_clarification → End | continue → build_context → guarded_generate → format_response → End)`.
</details>

### 2.1 Nodes and routing

| Node | Responsibility |
|------|----------------|
| `validate_input` | Extracts latest user message; calls Guardrails input rail `/v1/rails/check` (stage=input); sets `blocked` + `refusal_message` |
| `classify` | LLM 3-way router: `greeting` / `specification` / `question`; sets `applied_subspace` for specifications; T5 scattered gate pre-sets `needs_clarification` |
| `hybrid_classify` | Trained SVM-RBF classifier; escalates to Gemma via Guardrails when confidence below threshold (gated by `hybrid_classifier_enabled`); sets `applied_subspace` |
| `retrieve` | Rewrites query against history; KB `/search/api` with metadata filtering (`filter_path`); T1–T7 recall fixes; FAQ verification; EntityList supplement; DLQ |
| `generate_clarification` | LLM asks a short Persian disambiguating question listing the 4 subspaces; sets `finish_reason="clarify"` |
| `build_context` | Numbered sources (`[1]..[20]`) ≤14000 chars; system prompt (ICS credit scoring) + history + question; entity-list injection |
| `guarded_generate` | POST Guardrails `/v1/chat/completions` (model `gemma-4-31b-triton-trt`); maps `content_filter` → blocked |
| `format_response` | Returns `{content, finish_reason, citations}`; handles blocked, greeting, specification-ack, FAQ, G5 no-evidence refusal, faith/contradiction/subspace-mismatch guards |
| `format_refusal` | Reuses `format_response` (`graph.py` registers it as a second alias under that name) |

### 2.2 Graph wiring (`graph.py`)

```text
validate_input --(should_continue_after_validation)--> classify | format_refusal
classify       --(should_route_after_classify)-------> format_response | hybrid_classify | retrieve
hybrid_classify --> retrieve
retrieve       --(should_clarify)--------------------> generate_clarification | build_context
generate_clarification --> END
build_context --> guarded_generate --> format_response --> END
format_refusal --> END
format_response --> END
```

Conditional edges:

- `should_continue_after_validation(state)` → `"format_refusal"` if `blocked` else `"classify"`
- `should_route_after_classify(state)` → `"format_response"` if `classify in ("greeting","specification")` else `"hybrid_classify"` (when enabled) else `"retrieve"`
- `should_clarify(state)` → `"clarify"` if `needs_clarification` or the T5 scattered/ambiguity safety net fires; else `"continue"`

Supplemental entity rows short-circuit clarify: when the entity tool matched a dictionary, the model can answer directly, so clarification is suppressed.

Compiled with `MemorySaver` (dev checkpointer); exports `rag_graph` and `graph` (Studio entry point in `langgraph.json` → `studio_graph.py:graph`).

## 3. Libraries

Runtime dependencies (`pyproject.toml`):

| Library | Version |
|---------|---------|
| fastapi | >=0.115.0 |
| uvicorn[standard] | >=0.30.0 |
| pydantic | >=2.10.0 |
| pydantic-settings | >=2.5.0 |
| httpx | >=0.27.0 |
| langgraph | >=0.2.0 |
| langchain-core | >=0.3.0 |
| python-dotenv | >=1.0.0 |
| langgraph-checkpoint | >=2.0.0 |
| langfuse | >=2.60.0 |
| opentelemetry-api | >=1.24.0 |
| opentelemetry-sdk | >=1.24.0 |
| opentelemetry-exporter-otlp | >=1.24.0 |

Dev dependencies:

| Library | Version |
|---------|---------|
| pytest | >=8.0.0 |
| pytest-asyncio | >=0.23.0 |
| pytest-cov | >=5.0.0 |
| ruff | >=0.5.0 |
| mypy | >=1.10.0 |
| langgraph-cli[inmem] | >=0.3.0 |

Both HTTP clients (`KnowledgebaseClient`, `GuardrailsClient`) use `httpx.AsyncClient` with `trust_env=False` so localhost calls bypass the corporate Squid proxy.

## 4. API contract

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Process health `{"status":"ok"}` |
| `GET` | `/ready` | Dependency readiness for KB + Guardrails; ready only when both ready |
| `GET` | `/v1/models` | Model discovery (`ics-helper-agent`) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat entry point (streaming + non-streaming) |

<details>
<summary>Request / response shape</summary>

Request (`POST /v1/chat/completions`):
```json
{
  "model": "gemma-4-31b",
  "messages": [{"role":"user","content":"چگونه می‌توانم گزارش اعتباری خود را دریافت کنم؟"}],
  "max_tokens": 500,
  "temperature": 0.0,
  "stream": true,
  "filter_path": "گزارش اعتبارسنجی چک"
}
```

Non-streaming response includes `rag` metadata (`request_id`, `citations`, `applied_subspace`, `available_subspaces`, `supplemental`, `follow_ups`) and an optional `audit` trail. Streaming returns SSE `data:` chunks ending with `data: [DONE]`; the final chunk carries `rag` metadata and `finish_reason` (`stop`, `content_filter`, or `clarify`). Graph failure → `500 {"detail":"Orchestration failed"}`; unhandled exception → `500 {"detail":"Internal server error"}`.
</details>

The API layer restores sticky `applied_subspace` from the LangGraph `MemorySaver` checkpoint, and — because Open WebUI sends no `X-Request-ID` (new thread per turn) — mines sticky from the user-only chat history (interrogatives skipped, so only declarations set scope). T4 injection applies the sticky `filter_path` for generic follow-ups; B sticky decay drops the subspace after `sticky_decay_turns` (default 3) consecutive generic turns.

## 5. Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `ORCHESTRATOR_HOST` | `0.0.0.0` | Bind host |
| `ORCHESTRATOR_PORT` | `8100` | |
| `KB_BASE_URL` | `http://127.0.0.1:8000` | KB Manager |
| `GUARDRAILS_BASE_URL` | `http://127.0.0.1:8200` | Guardrails |
| `REQUEST_TIMEOUT_SECONDS` | `120` | Downstream timeout |
| `RETRIEVAL_TOP_K` | `20` | 10 RRF + 10 cross-encoder union |
| `SUBSPACE_KEYWORD_BOOST` | `{cheque:3.5, facilities_person:3.0, facilities_legal:3.0, general:2.5}` | Per-subspace BM25 bias on scoped calls |
| `SUBSPACE_MAP` | 4 entries | subspace label → KB `filter_path` |
| `STICKY_DECAY_TURNS` | `3` | Generic turns before sticky subspace is dropped |
| `EVIDENCE_MIN_SCORE` | `0.15` | Global no-evidence refusal floor |
| `RESPONSE_DETAIL` | `production` | `production` hides the retrieval block; `verbose` appends it |
| `UPSTREAM_LLM_MODEL` | `gemma-4-31b-triton-trt` | Judge/generate model |
| `HYBRID_CLASSIFIER_ENABLED` | `false` | Thresholded SVM→Gemma hybrid |
| `TRACE_ENABLED` | `true` | Langfuse tracing |

Per-subspace evidence thresholds (T3): `cheque 0.05`, `facilities_person 0.10`, `facilities_legal 0.15`, `general 0.15`.

## 6. State (`RAGState`)

Carried through the graph: `request_id`, `messages`, `query`, `rewritten_query`, `guardrail_decision`, `blocked`, `refusal_message`, `retrieved_chunks`, `prompt_messages`, `answer`, `citations`, `formatted_response`, `classify`, `applied_subspace`, `sticky_subspace`, `needs_clarification`, `clarifying_question`, `available_subspaces`, `filter_path`, `injected_entities`, `supplemental`, `greeting_only`, `direct_faq_answer`, `faq_matched`, `faq_verified`, `stage_timing_ms`, `fallback_used`, `fallback_reason`, `broaden_used`, `error`, `follow_ups`, plus ambiguity score fields (`ambiguity_score`, `ambiguity_margin`, `ambiguity_entropy`, `ambiguity_ambiguous`, `ambiguity_source`, `subspace_sims`).

`record_timing(state, stage, elapsed_ms)` centralizes stage-timing bookkeeping (`state.py`).

## 7. Tracing (Langfuse)

Every request creates a trace keyed by `X-Request-ID` with spans: `validate_input`, `classify`, `query-rewrite`, `retrieve` (+ `retrieve-supplement-fallback`, `retrieve-dlq`, `retrieve-retry`, `retrieve-broadened`), `build_context` (+ `compact`), `guarded_generate`, `format_response` / `generate_clarification`, and a final `trace-create` upsert with `output` and `metadata` (`citations`, `latency_ms`, `finish_reason`). Falls back to the local JSONL collector at `:3000` when Langfuse `:3001` is unavailable. View via `GET http://127.0.0.1:3000/observe/{request_id}` or Langfuse UI `:3001`.

## 8. Changelog (summary)

| Version | Summary |
|---------|---------|
| `v0.1.0` (current, production) | Deterministic LangGraph orchestrator: ambiguity resolver over 4 subspaces (`cheque`, `facilities_person`, `facilities_legal`, `general`), clarifying-question generator, metadata filtering (`filter_path`), sticky subspace with T4 injection and B sticky decay, T1–T7 recall fixes, G5 faith/contradiction/subspace-mismatch guards, grounded follow-ups, thresholded hybrid SVM→Gemma classifier, EntityList supplement + DLQ, Langfuse tracing, OpenAI-compatible streaming API |

## 9. Running locally & tests

```bash
cd components/orchestrator
PYTHONPATH=./src \
  KB_BASE_URL=http://127.0.0.1:8000 \
  GUARDRAILS_BASE_URL=http://127.0.0.1:8200 \
  /tmp/orch-venv/bin/python -m uvicorn work_rag_orchestrator.api:create_app --factory --host 0.0.0.0 --port 8100
```

```bash
cd components/orchestrator
PYTHONPATH=./src /tmp/orch-venv/bin/python -m pytest tests/ -v
```

## 10. Links

- [STUDIO.md](STUDIO.md) — local graph debugger guide (LangGraph Studio)
- [docs/SUBSPACE_AMBIGUITY.md](docs/SUBSPACE_AMBIGUITY.md) — subspace ambiguity score design spec (frozen interface)
- [Parent README](../README.md)

<!-- docs-toc:start -->
## Documentation Index

> Auto-generated from this repository's tracked Markdown notes (**3** files). Refresh with `python scripts/gen_readme_toc.py` — the `docs-toc` workflow does this automatically on push.

### Repository root

- [LangGraph Studio — Orchestrator](STUDIO.md) — LangGraph Studio is the visual debugger for the deterministic RAG graph:

### `docs/`

- [Subspace Ambiguity Score — Design Spec (frozen interface)](docs/SUBSPACE_AMBIGUITY.md) — Goal: replace brittle keyword-only ambiguity detection with a calibrated embedding

### `evaluation/`

- [Work Credit RAG — Benchmark Report](evaluation/artifacts/bench_report_latest.md) — Artifacts: bench_results_latest.json + ragas_results_latest.json → bench_report_latest.html / .md

<!-- docs-toc:end -->
