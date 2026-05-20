import sys
from pathlib import Path

# Ensure src/ is on sys.path so imports work when pytest is run from repo root
root = Path(__file__).resolve().parents[1]
src_path = str(root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from multi_agentic_crag.src.agents.router import route_query


def test_router_prefers_web_for_latest():
    state = {"user_message": "What is the latest on AI?", "faiss_index": None}
    assert route_query(state) == "web"
