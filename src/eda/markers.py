"""
exploring ssr data
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import matplotlib.pyplot as plt
import seaborn as sns

def go_back_dir(path, number):
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def process_group(group):
    if group.apply(lambda x: x.notna().all()).all():
        # All values are not NaN, check if they are equal
        if group.nunique().eq(1).all():
            # All values are equal, keep only one row
            return group.iloc[:1]
        else:
            # Values are not equal, add a running number to 'clone name'
            # for i in range(len(group)):
            #     group['Clone name'][i] = f"{group['Clone name'].iloc[0]} : {i + 1}"
            group.reset_index(inplace=True)
            group['Clone name'] = f"{group['Clone name'].iloc[0]} " + (group.index + 1).astype(str)
            return group
    else:
        # Keep all rows if any NaN values are present
        return group
    
def fillna_mode(group):
    return group.fillna(value=group.mode().iloc[0])

        
def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    
    # markers data
    ssr_info = pd.read_sql_query("SELECT * FROM ssr", connection)
    ssr_info.set_index(["Clone name", "Refcode"], inplace=True)
    # ssr_info.drop(columns=["Refcode"], inplace=True)
    # ssr_info.set_index(['Clone name'], inplace=True)
    numeric_ssr_info = ssr_info.apply(pd.to_numeric, errors='coerce')
    numeric_ssr_info.reset_index(inplace=True)
    grouped_ssr_info = numeric_ssr_info.groupby(["Clone name", "Refcode"])
    # grouped_ssr_info = numeric_ssr_info.groupby(['Clone name'])
    ssr_filled = grouped_ssr_info.apply(fillna_mode)
    # grouped_ssr_filled = ssr_filled.reset_index(drop=True).groupby(['Clone name'])
    # grouped_ssr_filled = ssr_filled.reset_index(drop=True).groupby(["Clone name", "Refcode"])
    ssr_result = ssr_filled.reset_index(drop=True).apply(fillna_mode)
    # ssr_result = grouped_ssr_filled.apply(process_group).reset_index(drop=True).apply(fillna_mode).drop(columns=["index"])
    ssr_result.set_index(["Clone name", "Refcode"], inplace=True)

    #################################################################

    filtered_df = ssr_result.reset_index().set_index("Clone name")
    filtered_df = filtered_df[filtered_df.index.str.contains('ICS')]

    # Calculate distances using pdist
    distances = pdist(filtered_df, metric='euclidean')
    distance_matrix = squareform(distances)
    linkage_matrix = linkage(distances, method='average')

    dendrogram(linkage_matrix, labels=filtered_df.index, orientation='top', distance_sort='ascending')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Clone name')
    plt.ylabel('Distance')
    plt.show()

    #################################################################

    # Calculate distances using pdist
    distances = pdist(ssr_result, metric='euclidean')
    distance_matrix = squareform(distances)
    linkage_matrix = linkage(distances, method='average')

    # plt.figure(figsize=(12, 6))
    tree = dendrogram(linkage_matrix, labels=ssr_result.index, orientation='top', distance_sort='ascending')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Clone name')
    plt.ylabel('Distance')
    plt.grid(False)
    plt.show()
    # plt.savefig("")

    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(ssr_result)

    num_clusters = 7
    # cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(ssr_result)
    
    k_values = range(2, 11)
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(ssr_result)
        silhouette_scores.append(silhouette_score(ssr_result, cluster_labels))

    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.grid(False)
    plt.show()

    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(ssr_result), columns=["pca1", "pca2"])
    df_pca.index = ssr_result.reset_index()["Clone name"]

    plt.scatter(df_pca['pca1'], df_pca['pca2'], c=cluster_labels, alpha=0.7, cmap='viridis', edgecolors='w', linewidth=0.5)
    plt.title('PCA Scatter Plot')
    plt.xlabel('Principal Component 1 (pca1)')
    plt.ylabel('Principal Component 2 (pca2)')
    plt.grid(False)
    for clone_name, pc1, pc2 in zip(df_pca.index, df_pca['pca1'], df_pca['pca2']):
        if clone_name in filtered_df.index:
            plt.annotate(clone_name, (pc1, pc2), textcoords="offset points", xytext=(-5,5), ha='right')
    plt.show()

    #################################################################

    query = """
        SELECT
            "bean"."Clone name",
            "bean"."Refcode",
            "bean"."Total dry weight",
            "butterfat"."Fat",
            "yield"."Yield ha"
        FROM
            "bean"
        JOIN
            "butterfat" ON "bean"."Refcode" = "butterfat"."Refcode" AND "bean"."Clone name" = "butterfat"."Clone name"
        JOIN
            "yield" ON "bean"."Refcode" = "yield"."Refcode" AND "bean"."Clone name" = "yield"."Clone name";
    """

    bean_fat_info = pd.read_sql_query(query, connection)

    # bean_fat_info.drop(columns=["Refcode"], inplace=True)
    # bean_fat_info.set_index(['Clone name'], inplace=True)
    # numeric_bean_fat_info = bean_fat_info.apply(pd.to_numeric, errors='coerce')
    # numeric_bean_fat_info.reset_index(inplace=True)
    # grouped_bean_fat_info = numeric_bean_fat_info.groupby(['Clone name'])
    # bean_fat_filled = grouped_bean_fat_info.apply(fillna_mode)
    # grouped_bean_fat_filled = bean_fat_filled.reset_index(drop=True).groupby(['Clone name'])
    # bean_fat_result = grouped_bean_fat_filled.apply(process_group).reset_index(drop=True).apply(fillna_mode).drop(columns=["index"])
    # bean_fat_result.set_index(['Clone name'], inplace=True)


    columns_to_process = ["Total dry weight", "Fat", "Yield ha"]

    bean_fat_info.set_index(["Clone name", "Refcode"], inplace=True)
    bean_fat_info.replace({'-' : None}, inplace=True)
    numeric_bean_fat_info = bean_fat_info.apply(pd.to_numeric, errors='coerce')
    median_values = numeric_bean_fat_info.median()
    bean_fat_result = numeric_bean_fat_info.apply(lambda col: col.fillna(median_values[col.name]))

    # spearman_corr = numeric_bean_fat_info_filled.corr(method='spearman')

    merged_result = pd.merge(bean_fat_result, ssr_result, left_index=True, right_index=True, how='inner')

    ############################################################

    sns.set(font_scale=0.8)
    sns.set_style("whitegrid")

    # Plot dendrogram
    dendrogram(linkage_matrix, no_labels=True, above_threshold_color='gray', color_threshold=0.7 * linkage_matrix.max())

    # Create a clustermap
    clustermap = sns.clustermap(result_df, row_linkage=linkage_matrix, col_cluster=False, cmap='viridis', figsize=(10, 8))

    # Add annotations to dendrogram branches
    for i, label in enumerate(result_df.index):
        clustermap.ax_row_dendrogram.bar(0, 0, color=clustermap.row_colors[i], label=label)

    plt.subplots_adjust(right=0.8)
    cax = plt.axes([0.85, 0.15, 0.02, 0.7])
    plt.colorbar(cax=cax, label='Phenotypic Value')
    plt.suptitle('Spearman Correlation Heatmap and Dendrogram')
    plt.show()

    ############################################################

    connection.close()

if __name__ == "__main__":
    main()

