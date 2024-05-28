'''
Order clones by centrality measurment
'''

import os
import sys
import pandas as pd
import networkx as nx
from build_pedigree_graph import build_graph

sys.path.append("/home/public-cocoa/src/")
from path_utils import go_back_dir

def order_by_centraliry(graph):
    degree_centrality = nx.degree_centrality(graph)
    ordered_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)
    return ordered_nodes

def nacodes_to_clonenames(nacodes):
    clonenames = []
    for node in nacodes:
        matching_rows = collection[collection['NA Code'] == node]
        if not matching_rows.empty:
            clonenames.append(matching_rows['Clone Name'].values[0])
        else:
            clonenames.append(f"Unknown Clone {node}")
    return clonenames

# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 2)
    pedigree_file = os.path.join(project_path, "data", "pedigree.json")
    graph = build_graph(pedigree_file)
    ordered_nodes = order_by_centraliry(graph)
    collection_file = os.path.join(project_path, "data", "cocoa_collection.csv")
    collection = pd.read_csv(collection_file, dtype=str)
    # Map nacodes to clone names
    ordered_clone_names = nacodes_to_clonenames(ordered_nodes)

    # Print the ordered clone names
    print("Ordered Clones by Degree Centrality:")
    for clone_name in ordered_clone_names[0:10]:
        print(clone_name)
