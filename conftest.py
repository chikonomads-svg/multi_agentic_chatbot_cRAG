"""Top-level pytest conftest to ensure project packages are importable during tests."""
import sys
from pathlib import Path


def pytest_configure(config):
    # Add repository root to sys.path so old_project and multi_agentic_crag packages
    # can be imported during pytest collection regardless of current working dir.
    repo_root = Path(__file__).resolve().parents[0]
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    # Also add known package parent directories so tests that import 'src.*'
    # from different subprojects resolve correctly when running pytest at repo root.
    candidates = [repo_root / 'multi_agentic_crag', repo_root / 'old_project' / 'rag-chatbot-app']
    for c in candidates:
        cstr = str(c)
        if c.exists() and cstr not in sys.path:
            sys.path.insert(0, cstr)
