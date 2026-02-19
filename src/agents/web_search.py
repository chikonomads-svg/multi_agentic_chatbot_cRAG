"""Web Search Agent: uses Tavily to fetch web results as fallback."""
from typing import Dict, Any, List
import logging
import os
from tavily import TavilyClient

LOG = logging.getLogger(__name__)


def tavily_search(query: str, api_key: str, max_results: int = 5) -> List[dict]:
    # Prefer the provided api_key (from state) but fall back to environment
    # variable TAVILY_API_KEY so the app can be configured via .env only.
    key = api_key or os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=key)
    try:
        resp = client.search(query, max_results=max_results)
        return resp.get("results", [])
    except Exception:
        LOG.exception("Tavily search failed for query=%s", query)
        return []


def search_web(state: Dict[str, Any]) -> List[dict]:
    key = state.get("tavily_api_key")
    if not key:
        LOG.debug("No Tavily key configured; returning empty results")
        return []
    LOG.debug("Performing Tavily search for query=%s", state.get("user_message", ""))
    return tavily_search(state.get("user_message", ""), api_key=key)
