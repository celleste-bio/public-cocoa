"""
exploring ssr data
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def go_back_dir(path, number):
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    
    # fat data
    ssr_info = pd.read_sql_query("SELECT * FROM ssr", connection)
    numeric_ssr_info = ssr_info.apply(pd.to_numeric, errors='coerce')
    numeric_ssr_info['Clone name'] = ssr_info['Clone name']
    grouped_ssr_info = numeric_ssr_info.groupby('Clone name').mean()
    grouped_ssr_info.drop(columns=["Refcode"], inplace=True)
    grouped_ssr_info = grouped_ssr_info.apply(lambda x: x.fillna(x.mean()))

    distances = pdist(grouped_ssr_info, metric='euclidean')
    distance_matrix = squareform(distances)
    linkage_matrix = linkage(distances, method='average')

    num_clusters = 10
    cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    # plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=grouped_ssr_info.index, orientation='top', distance_sort='ascending')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Clone name')
    plt.ylabel('Distance')
    plt.show()
    # plt.savefig("")

    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(grouped_ssr_info), columns=["pca1", "pca2"])
    df_pca.index = grouped_ssr_info.index

    plt.scatter(df_pca['pca1'], df_pca['pca2'], c=cluster_labels, cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.title('PCA Scatter Plot')
    plt.xlabel('Principal Component 1 (pca1)')
    plt.ylabel('Principal Component 2 (pca2)')
    plt.colorbar(label='Cluster')
    plt.grid(False)
    for clone_name, pc1, pc2 in zip(df_pca.index, df_pca['pca1'], df_pca['pca2']):
        if clone_name in ["ICS 60", "ICS 95", "ICS 6", "ICS 14"]:
            plt.annotate(clone_name, (pc1, pc2), textcoords="offset points", xytext=(-5,5), ha='right')
    plt.show()
    
if __name__ == "__main__":
    main()

