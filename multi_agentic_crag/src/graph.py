from typing import TypedDict, Optional, List
import logging
import os

"""Provide a minimal Graph/Node implementation used by the app.

Prefer a langgraph-provided Graph/Node if available, but to avoid
import-time failures this module defines a lightweight local Graph and
Node API with the subset of behavior the app requires.

This keeps imports simple (no try/except around imports) and ensures the
app runs even when langgraph's public exports differ between versions.
"""

from typing import Callable, Dict, Any


class Node:
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func


class Graph:
    def __init__(self, name: str = "fallback-graph"):
        self._name = name
        self._nodes: Dict[str, Node] = {}

    def add_node(self, node: Node) -> None:
        self._nodes[node.name] = node

    def run_node(self, node_name: str, *args, **kwargs):
        node = self._nodes.get(node_name)
        if not node:
            raise KeyError(f"Node not found: {node_name}")
        # Call the underlying function. The small agents expect (state, ...)
        return node.func(*args, **kwargs)


from multi_agentic_crag.src.agents.router import route_query
from multi_agentic_crag.src.agents.retriever import retrieve_local
from multi_agentic_crag.src.agents.grader import grade_chunks
from multi_agentic_crag.src.agents.web_search import search_web
from multi_agentic_crag.src.agents.analyst import detect_and_build_plot
from multi_agentic_crag.src.agents.generator import generate_answer

LOG = logging.getLogger(__name__)


class State(TypedDict, total=False):
    user_message: str
    faiss_index: Optional[str]
    tavily_api_key: Optional[str]
    retrieved: Optional[List]
    plot_json: Optional[dict]
    active_agent: Optional[str]


def build_graph() -> Graph:
    g = Graph(name="multi-agent-crag")

    # Nodes are simple functional wrappers; langgraph usage here is illustrative
    g.add_node(Node("router", func=route_query))
    g.add_node(Node("retriever", func=retrieve_local))
    g.add_node(Node("grader", func=grade_chunks))
    g.add_node(Node("web_search", func=search_web))
    g.add_node(Node("analyst", func=detect_and_build_plot))
    g.add_node(Node("generator", func=generate_answer))

    return g
