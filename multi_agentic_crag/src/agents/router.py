"""Router Agent: decides whether to use FAISS, web search, or direct computation."""
from typing import Dict, Any
import logging

LOG = logging.getLogger(__name__)


def route_query(state: Dict[str, Any]) -> str:
    """Very small heuristic router: if the query contains 'latest' or '202' prefer web search.

    Otherwise if a faiss_index is present in state, prefer local retrieval.
    Returns: one of 'local', 'web', 'calc'
    """
    query = state.get("user_message", "").lower()
    if any(tok in query for tok in ["latest", "recent", "202", "news"]):
        LOG.debug("Router: routing to web search for query=%s", query)
        return "web"
    if state.get("faiss_index"):
        LOG.debug("Router: routing to local retrieval for query=%s", query)
        return "local"
    LOG.debug("Router: routing to web search fallback for query=%s", query)
    return "web"
