# Work RAG Orchestrator

LangGraph orchestration and public OpenAI-compatible chat API for the Work Credit RAG platform.

> **Status: implemented and operational.** The deterministic RAG graph runs end-to-end and is also loadable in LangGraph Studio.

## 1. Summary

The Orchestrator owns **request coordination**. It receives a chat completion request, runs it through a deterministic LangGraph workflow, and returns a grounded, citation-shaped response.

Request path:

```text
browser/Open WebUI
  └─▶ Orchestrator POST /v1/chat/completions
        ├─▶ Guardrails input check        (validate_input)
        ├─▶ KB hybrid retrieval           (retrieve)
        ├─▶ bounded context build         (build_context)
        ├─▶ Guardrails guarded generation (guarded_generate → Gemma)
        ├─▶ OpenAI response + citations   (format_response)
        └─▶ browser
```

It does **not** own KB ingestion/retrieval algorithms, safety policy definitions, model lifecycle, or the frontend.

---

## 2. Graph architecture

### 2.1 Nodes and routing

```mermaid
flowchart TD
    START([START]) --> VI[validate_input<br/>Guardrails /v1/rails/check stage=input]
    VI --> C{guardrail_decision.allowed?}
    C -->|yes| RET[retrieve<br/>KB POST /search/api top_k=5]
    C -->|no / no user msg| REF[format_refusal<br/>reuses format_response]
    RET --> BC[build_context<br/>[1..5], ≤8000 chars, system+user prompt]
    BC --> GG[guarded_generate<br/>Guardrails /v1/chat/completions]
    GG --> FR[format_response<br/>content + citations + finish_reason]
    FR --> STOP([END])
    REF --> STOP
```

### 2.2 Node responsibilities

| Node | Input → Output | Detail |
|---|---|---|
| `validate_input` | messages → query, guardrail_decision | Extracts **latest** user message; calls Guardrails input check; sets `blocked` + `refusal_message` |
| `retrieve` | query → retrieved_chunks | POST `/search/api` with `top_k=5`, `X-Request-ID`; normalizes KB `final_results` |
| `build_context` | chunks → prompt_messages | Numbered `[1]..[5]` sources (title/heading/content); truncates at 8000 chars; adds system prompt (ICS credit scoring) + `Question: ...` |
| `guarded_generate` | prompt_messages → answer | POST Guardrails `/v1/chat/completions` (model `gemma-4-31b`); maps `content_filter` → blocked |
| `format_response` | answer/chunks → formatted_response | Returns `{content, finish_reason, citations}`; blocked → `finish_reason="content_filter"` |

The `format_refusal` node **reuses** `format_response` (`graph.py` adds it a second time under that alias).

### 2.3 Graph wiring (`graph.py`)

```text
validate_input --(conditional)--> retrieve | format_refusal
retrieve --> build_context --> guarded_generate --> format_response --> END
format_refusal --> END

Conditional: should_continue_after_validation(state) -> "retrieve" | "format_refusal"
Compiled with MemorySaver (dev checkpointer); exports `rag_graph` and `graph` (Studio).
```

---

## 3. Clients (downstream adapters)

### GuardrailsClient

| Method | Call | Fail behaviour |
|---|---|---|
| `check_rails(stage, text, request_id)` | POST `/v1/rails/check` | **Fail closed** → `{allowed: false, categories: ["engine_error"|"timeout"]}` |
| `chat_completion(request, request_id)` | POST `/v1/chat/completions` (`X-Request-ID` header) | Re-raises on error |
| `health_check()` | GET `/health` | bool |
| `readiness_check()` | GET `/ready` | dict |

### KnowledgebaseClient

| Method | Call | Mapping |
|---|---|---|
| `retrieve(query, top_k, request_id)` | POST `/search/api` | KB `final_results[]` → `KBRetrievalResult`: `chunk_id`, `doc_id→document_id`, `doc_title→title`, `heading_path→heading`, `content_preview→content`, `rerank_score`/`hybrid_score→score` |
| `health_check()` | GET `/` (KB has no `/health`) | bool |

> Both clients use `http(s) ASTransport` with `trust_env=False` deliberately — localhost calls must **bypass the corporate proxy** (Squid) so they don't route through the internet.

---

## 4. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process liveness `{"status":"ok"}` |
| `GET` | `/ready` | Dependency readiness for KB + Guardrails → `{status, dependencies[]}`; ready only when both ready |
| `POST` | `/v1/chat/completions` | OpenAI-compatible non-streaming chat; graph runs; returns `ChatCompletionResponse` + `rag` metadata |

### Chat request / response

```jsonc
// Request  (POST /v1/chat/completions)
{
  "model": "gemma-4-31b",
  "messages": [{"role":"user","content":"چگونه می‌توانم گزارش اعتباری خود را دریافت کنم؟"}],
  "max_tokens": 500,
  "temperature": 0.0
}

// Response
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "gemma-4-31b",
  "choices": [{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
  "usage": null,
  "rag": {
    "request_id": "6674c17f-...",
    "citations": [{"chunk_id":"...","document_id":"...","title":"...","heading":"..."}]
  }
}
```

Behavior notes:

- `X-Request-ID` (or a generated UUID) becomes the tracing ID and the LangGraph `thread_id`.
- Non-`gemma-4-31b` models are logged (warning) but not rejected.
- Graph failure → `500 {"detail":"Orchestration failed"}`; unhandled exception → `500 {"detail":"Internal server error"}`.

---

## 5. State (`RAGState`)

```python
class RAGState(TypedDict):
    request_id: str
    messages: list[dict[str, str]]
    query: str
    guardrail_decision: dict | None
    retrieved_chunks: list[dict]
    prompt_messages: list[dict[str, str]]
    answer: str
    citations: list[dict]
    error: str | None
    blocked: bool
    refusal_message: str | None
    formatted_response: dict | None          # consumed by API layer
    # Phase 3+ fields (unused in MVP)
    retrieve_attempts: int
    graded_chunks: list[dict]
    hallucination_spans: list[str]
    grounded: bool
```

---

## 6. Configuration (`.env.example`)

```dotenv
ORCHESTRATOR_PORT=8100
KB_BASE_URL=http://127.0.0.1:8000
GUARDRAILS_BASE_URL=http://127.0.0.1:8200
REQUEST_TIMEOUT_SECONDS=120
RETRIEVAL_TOP_K=5
```

Computed downstream URLs:

| Property | Value |
|---|---|
| `kb_search_url` | `{KB_BASE_URL}/search/api` |
| `guardrails_rails_check_url` | `{GUARDRAILS_BASE_URL}/v1/rails/check` |
| `guardrails_chat_url` | `{GUARDRAILS_BASE_URL}/v1/chat/completions` |

Copy to `.env` before running.

---

## 7. Run

```bash
cd components/orchestrator
cp .env.example .env
pip install -e ".[dev]"
python -m work_rag_orchestrator.api            # port 8100
# or: orchestrator                                   (console script)
# or: docker build -t work-rag-orchestrator . && docker run -p 8100:8100 work-rag-orchestrator
```

### LangGraph Studio (visual debugging)

```bash
pip install -e ".[dev]"      # installs langgraph-cli[inmem]
langgraph dev --port 8120    # open http://localhost:8120  -> graph "rag"
```

`langgraph.json` → `graphs.rag = ./studio_graph.py:graph`. Studio gives you the interactive canvas, per-node state, thread history, and timing. Stressed in [`STUDIO.md`](STUDIO.md).

---

## 8. Tests

`tests/test_orchestrator.py`:

- `validate_input`: extracts latest user message; blocks on guardrails refusal
- `retrieve`: calls KB and normalizes results
- `build_context`: numbered `[1]`/`[2]` markers + system/user messages
- `guarded_generate`: forwards to guardrails chat; stores answer / blocked
- `format_response`: allowed → citations + `stop`; blocked → `content_filter` + empty citations
- Graph structure: all 6 nodes present

```bash
pytest tests/ -v
```

---

## 9. Planning & progress checklist

### Done (MVP)

- [x] Deterministic RAG graph with conditional refusal branch
- [x] `validate_input` → Guardrails input rail check (fail-closed upstream behaviour)
- [x] `retrieve` → KB `/search/api` with field normalization
- [x] `build_context` → bounded numbered context (≤5 chunks, 8000 chars)
- [x] `guarded_generate` → Guardrails chat gateway (modal `gemma-4-31b`)
- [x] `format_response` → citations + `finish_reason`; refusal path reuses node
- [x] OpenAI-compatible `/v1/chat/completions` + `rag` metadata (request_id, citations)
- [x] `/health`, `/ready` with dependency checks
- [x] LangGraph Studio integration (`studio_graph.py`, `langgraph.json`)
- [x] Dockerfile, `.env.example`, node unit tests

### Next / open

- [ ] API-level tests for `/health`, `/ready`, `/v1/chat/completions`
- [ ] Contract tests validating `orchestrator_chat_response.json` schema
- [ ] PostgreSQL checkpointing (AsyncPostgresSaver) for durable thread memory
- [ ] Retrieval retry loop with graded chunks (`grade_docs`)
- [ ] Hallucination span detection (ISSUP/ISUSE-style nodes)
- [ ] Query rewriting / multi-query generation wired end-to-end
- [ ] Streaming responses (`stream=true`)
- [ ] Reconcile STUDIO.md reference (`graph.py:graph`) with actual `studio_graph.py:graph`

---

## License

See parent repository `LICENSE` and the submodule's own obligations.