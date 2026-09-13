# LangGraph Studio — Orchestrator

LangGraph Studio is the visual debugger for the deterministic RAG graph `validate_input -> retrieve -> build_context -> guarded_generate -> format_response`.

## Setup

```bash
cd components/orchestrator
pip install -e ".[dev]"  # installs langgraph-cli[inmem]

# .env already has KB_BASE_URL, GUARDRAILS_BASE_URL
cp .env.example .env

langgraph dev --port 8120
# open http://localhost:8120  -> graph "rag"
```

Studio auto-loads `langgraph.json:graphs.rag = ./src/work_rag_orchestrator/graph.py:graph`.

## What you get

- Interactive graph canvas with 6 nodes + conditional edge `validate_input -- blocked --> format_refusal`
- Per-node input/output state (`RAGState`) with Persian `query`, `retrieved_chunks`, `citations`
- Thread history (MemorySaver) - replay same `thread_id` (= `request_id`)
- Time per node, hit RRF/reranker scores
- Test with Persian examples directly in Studio's chat pane

## Modifying in Studio

1. Edit `src/work_rag_orchestrator/nodes/*.py` or `state.py`
2. Studio hot-reloads (no restart). For `graph.py` structural changes, restart `langgraph dev`.
3. Add a node: `graph.add_node("query_rewrite_fa", ...)` in `graph.py:25`

No separate server UI is needed for prod - Studio is dev-only. Prod runs `python -m work_rag_orchestrator.api` on :8100.
