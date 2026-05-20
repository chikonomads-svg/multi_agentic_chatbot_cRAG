"""Retriever Agent: interfaces with the FAISS vector store and retrieves top-k chunks."""
from typing import Any, Dict, List, Tuple
import logging

# Use Document class from our vector_store module which provides a fallback
# when langchain is not available.
from ..vector_store import load_faiss, search_faiss, Document
import logging

LOG = logging.getLogger(__name__)

LOG = logging.getLogger(__name__)


def retrieve_local(state: Dict[str, Any], k: int = 5) -> List[Tuple[Document, float]]:
    """Retrieve top-k documents from a loaded FAISS index in the state.

    Expects state['faiss_index'] to be a path string or a loaded index object.
    Returns list of (Document, score).
    """
    idx = state.get("faiss_index")
    if isinstance(idx, str):
        LOG.debug("Loading FAISS index from %s", idx)
        idx = load_faiss(idx)
        state["faiss_index"] = idx
    if idx is None:
        LOG.debug("No FAISS index available in state")
        return []
    results = search_faiss(idx, state.get("user_message", ""), k=k)
    return results
