"""LangGraph graph definition for RAG orchestration."""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
# O-C3: MemorySaver is unbounded — for prod use Postgres/Redis checkpointer or TTL. See state.py record_timing for leak mitigation.
# For now, limit to 100 threads via custom wrapper (evict oldest)

from .state import RAGState
from .subspaces import SUBSPACES
from .nodes import (
    validate_input,
    retrieve,
    build_context,
    guarded_generate,
    format_response,
    hybrid_classify_node,
)
from .nodes.generate_clarification import generate_clarification
from .nodes.classify import classify


def should_continue_after_validation(state: RAGState) -> str:
    """Conditional edge: if blocked, go to format_refusal, else continue to classify."""
    if state.get("blocked", False):
        return "format_refusal"
    return "classify"


def should_route_after_classify(state: RAGState) -> str:
    """Conditional edge after LLM classify: greeting/spec go directly to format_response, question -> hybrid_classify."""
    cat = state.get("classify")
    if cat in ("greeting", "specification"):
        return "format_response"
    return "hybrid_classify"


def should_clarify(state: RAGState) -> str:
    """Branch after retrieve: clarify vs continue.

    T5: ensure scattered ambiguous queries with no clear subspace always clarify,
    even if retrieve's threshold/general_exempt suppressed the flag. This is a
    safety net in addition to retrieve's early scattered detection.
    """
    if state.get("needs_clarification"):
        return "clarify"
    # Stage 3 (route): the embedding ambiguity score can force clarification
    # even when the keyword safety net below does not fire — only when no
    # explicit/sticky subspace resolved. Gated by subspace_ambiguity_route so
    # default behavior is unchanged; settings imported locally to avoid an
    # import cycle at module load.
    try:
        from .config import get_settings

        if (
            get_settings().subspace_ambiguity_route
            and state.get("ambiguity_ambiguous")
            and not state.get("applied_subspace")
        ):
            if not state.get("available_subspaces"):
                state["available_subspaces"] = list(SUBSPACES)
            state["needs_clarification"] = True
            return "clarify"
    except Exception:
        pass
    # Safety net: if applied_subspace is None and query is scattered ambiguous, force clarify
    try:
        from .nodes.infer_specification import is_scattered_ambiguous

        q = state.get("query") or ""
        # Only trigger when no subspace was resolved and query looks scattered
        if not state.get("applied_subspace") and is_scattered_ambiguous(q):
            # Also ensure we have available_subspaces for the clarification node
            if not state.get("available_subspaces"):
                state["available_subspaces"] = list(SUBSPACES)
            state["needs_clarification"] = True
            return "clarify"
    except Exception:
        pass
    return "continue"


def create_graph() -> StateGraph:
    """Create the deterministic RAG graph."""
    
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("validate_input", validate_input)
    graph.add_node("classify", classify)
    graph.add_node("hybrid_classify", hybrid_classify_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("guarded_generate", guarded_generate)
    graph.add_node("format_response", format_response)
    graph.add_node("format_refusal", format_response)  # Reuse format_response for refusal
    graph.add_node("generate_clarification", generate_clarification)
    
    # Set entry point
    graph.set_entry_point("validate_input")
    
    # Conditional edge from validate_input -> classify (or refusal)
    graph.add_conditional_edges(
        "validate_input",
        should_continue_after_validation,
        {
            "classify": "classify",
            "format_refusal": "format_refusal",
        },
    )
    
    # Conditional edge from classify: greeting/spec -> format_response, question -> hybrid_classify
    graph.add_conditional_edges(
        "classify",
        should_route_after_classify,
        {
            "format_response": "format_response",
            "hybrid_classify": "hybrid_classify",
        },
    )
    
    # Hybrid classify -> retrieve (Mizan subspace continuity removed)
    graph.add_edge("hybrid_classify", "retrieve")
    
    # Conditional after retrieve: clarify vs build_context (Mizan retrieval judge removed)
    graph.add_conditional_edges(
        "retrieve",
        should_clarify,
        {
            "clarify": "generate_clarification",
            "continue": "build_context",
        },
    )
    
    graph.add_edge("generate_clarification", END)
    graph.add_edge("build_context", "guarded_generate")
    graph.add_edge("guarded_generate", "format_response")
    graph.add_edge("format_response", END)
    graph.add_edge("format_refusal", END)
    
    return graph


def compile_graph():
    """Compile the graph with memory saver for development."""
    graph = create_graph()
    # Use MemorySaver for development (no PostgreSQL checkpointing in MVP)
    return graph.compile(checkpointer=MemorySaver())


# Compiled graph instances
rag_graph = compile_graph()
# LangGraph Studio entry - must be named `graph` for langgraph.json
graph = rag_graph