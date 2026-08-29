# Work RAG Orchestrator

LangGraph orchestration and public chat API for the Work Credit RAG platform.

> Status: repository initialized for submodule integration. The runtime described below is the MVP contract to implement; it is not implemented yet.

## Responsibility

This component owns request coordination. It does not own KB maintenance/retrieval algorithms, NeMo policy definitions, model lifecycle, or the frontend.

The MVP graph is intentionally deterministic:

```text
START
  -> validate_input
  -> retrieve
  -> build_context
  -> guarded_generate
  -> format_response
  -> END
```

`validate_input` calls Work RAG Guardrails, `retrieve` calls Work RAG KB, and `guarded_generate` calls the Guardrails OpenAI-compatible gateway, which in turn calls the self-hosted Gemma manager in Work RAG Server Setup.

## MVP API contract

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process health |
| `GET` | `/ready` | Dependency readiness for KB and Guardrails |
| `POST` | `/v1/chat/completions` | OpenAI-compatible non-streaming chat entry point |

Initial configuration:

```dotenv
ORCHESTRATOR_PORT=8100
KB_BASE_URL=http://127.0.0.1:8000
GUARDRAILS_BASE_URL=http://127.0.0.1:8200
REQUEST_TIMEOUT_SECONDS=120
RETRIEVAL_TOP_K=5
```

## Planned structure

```text
.
├── README.md
├── pyproject.toml
├── .env.example
├── src/work_rag_orchestrator/
│   ├── api.py
│   ├── config.py
│   ├── graph.py
│   ├── state.py
│   ├── schemas.py
│   ├── nodes/
│   └── clients/
│       ├── guardrails.py
│       └── knowledgebase.py
└── tests/
    ├── contract/
    ├── integration/
    └── unit/
```

## Initial state contract

The first graph should carry only what the end-to-end path needs:

```python
class RAGState(TypedDict):
    request_id: str
    messages: list[dict[str, str]]
    query: str
    guardrail_decision: dict
    retrieved_chunks: list[dict]
    prompt_messages: list[dict[str, str]]
    answer: str
    citations: list[dict]
    error: str | None
```

No retries, memory, query rewriting, tool selection, or agent loops belong in the first graph. Add them after the deterministic path passes the parent repository's MVP acceptance suite.

See the parent repository's `docs/MVP_INTEGRATION_PLAN.md` for the cross-component implementation sequence and acceptance criteria.
