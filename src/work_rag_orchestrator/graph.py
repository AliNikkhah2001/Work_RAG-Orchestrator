"""LangGraph graph definition for RAG orchestration."""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import RAGState
from .nodes import (
    validate_input,
    retrieve,
    build_context,
    guarded_generate,
    format_response,
)


def should_continue_after_validation(state: RAGState) -> str:
    """Conditional edge: if blocked, go to format_refusal, else continue to retrieve."""
    if state.get("blocked", False):
        return "format_refusal"
    return "retrieve"


def create_graph() -> StateGraph:
    """Create the deterministic RAG graph."""
    
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("validate_input", validate_input)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("guarded_generate", guarded_generate)
    graph.add_node("format_response", format_response)
    graph.add_node("format_refusal", format_response)  # Reuse format_response for refusal
    
    # Set entry point
    graph.set_entry_point("validate_input")
    
    # Conditional edge from validate_input
    graph.add_conditional_edges(
        "validate_input",
        should_continue_after_validation,
        {
            "retrieve": "retrieve",
            "format_refusal": "format_refusal",
        },
    )
    
    # Linear path for allowed requests
    graph.add_edge("retrieve", "build_context")
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