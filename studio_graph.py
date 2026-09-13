"""Shim for LangGraph Studio - adds src to path, no custom checkpointer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from work_rag_orchestrator.graph import create_graph

# LangGraph API handles persistence; don't pass MemorySaver
graph = create_graph().compile()

__all__ = ["graph"]
