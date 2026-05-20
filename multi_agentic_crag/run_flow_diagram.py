import sys
import os

# Explicitly add the project root to sys.path to ensure module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from multi_agentic_crag.language_flow_diagram import create_graph_visualization

if __name__ == "__main__":
    create_graph_visualization()