'''
Build and visalise graph from pedigree.json file
'''

import os
import json
import networkx as nx
import matplotlib.pyplot as plt

def build_graph(pedigree_file):
    # Load JSON data
    with open(pedigree_file, 'r') as file:
        data = json.load(file)

    # Create a graph
    graph = nx.Graph()

    for node, neighbors in data.items():
        graph.add_node(node)
        for neighbor in neighbors:
            graph.add_edge(node, neighbor)

    return graph

def vis_graph(graph, output_file):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=False, font_weight='bold', node_size=2, node_color='blue', edge_color='gray')
    plt.savefig(output_file, format='pdf')


# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    pedigree_file = os.path.join(project_path, "data", "pedigree.json")
    graph = build_graph(pedigree_file)

    output_file = os.path.join(project_path, "data", "pedigree.pdf")
    vis_graph(graph, output_file)