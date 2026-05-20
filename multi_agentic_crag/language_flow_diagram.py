import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from graph import build_graph
import graphviz

# Define a function to visualize the graph using Graphviz
def create_graph_visualization(output_path="flow_diagram.gv"):
    g = graphviz.Digraph("LangraphFlow", format="png")

    graph = build_graph()
    nodes = graph._nodes
    for node_name, node in nodes.items():
        g.node(node_name)
        # Default edge just to illustrate flow
        g.edge("START", node_name)

    g.render(directory=".", filename=output_path, view=True)

if __name__ == "__main__":
    create_graph_visualization()
