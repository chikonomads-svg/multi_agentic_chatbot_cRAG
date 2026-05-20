import sys
from pathlib import Path


def pytest_configure(config):
    # Ensure the package 'src' is importable by adding the project root to sys.path
    root = Path(__file__).resolve().parents[1]
    project_root = str(root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
