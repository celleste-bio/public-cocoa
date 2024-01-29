"""
connection between:
* geo group -> continent
* geno group -> kmean on ssr validated visualy with pca
* pheno group -> fat precentage binned
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns


def go_back_dir(path, number):
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def fillna_mode(group):
    return group.fillna(value=group.mode().iloc[0])
    
def getgeo_groups(connection):
    query = "SELECT [Clone name], [Country] FROM planting"
    planting_info = pd.read_sql_query(query, connection)
    planting_info.dropna(inplace=True)
    planting_info["Clone name"].unique()
    planting_info.set_index(["Clone name"], inplace=True)
    country_continent_dict = {
        "Cote d'Ivoire": "Africa",
        "Indonesia": "Asia",
        "Dominican Republic": "North America",
        "Ecuador": "South America",
        "Brazil": "South America",
        "PAN,CRI,NIC,HND,GTM,BLZ": "North America",
        "Malaysia": "Asia",
        "Philippines": "Asia",
        "Trinidad and Tobago": "North America",
        "Peru": "South America"
    }

    planting_info.replace(country_continent_dict, inplace=True)
    planting_info.rename(columns={'Country': 'Continent'}, inplace=True)
    return planting_info
    

def get_geno_groups(connection):
    query = "SELECT * FROM ssr"
    ssr_info = pd.read_sql_query(query, connection)
    ssr_info.drop(columns=["Refcode"], inplace=True)
    ssr_info.set_index(["Clone name"], inplace=True)
    numeric_ssr_info = ssr_info.apply(pd.to_numeric, errors='coerce')
    numeric_ssr_info.reset_index(inplace=True)
    grouped_ssr_info = numeric_ssr_info.groupby(["Clone name"])
    ssr_filled = grouped_ssr_info.apply(fillna_mode)
    ssr_result = ssr_filled.reset_index(drop=True).apply(fillna_mode)
    ssr_result.set_index(["Clone name"], inplace=True)

    num_clusters = 6
    # cluster_labels = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(ssr_result)
    
    k_values = range(2, 11)
    silhouette_scores = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(ssr_result)
        silhouette_scores.append(silhouette_score(ssr_result, cluster_labels))

    # plt.plot(k_values, silhouette_scores, marker='o')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Analysis for Optimal k')
    # plt.show()
        
    silhouette_scores
    min_score_index = silhouette_scores.index(min(silhouette_scores))
    optimal_k = min_score_index + k_values[0]

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(ssr_result)

    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(ssr_result), columns=["pca1", "pca2"])
    df_pca.index = ssr_result.index
    df_pca["cluster"] = cluster_labels

    plt.scatter(df_pca['pca1'], df_pca['pca2'], c=df_pca["cluster"], alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.title('PCA Scatter Plot')
    plt.xlabel('Principal Component 1 (pca1)')
    plt.ylabel('Principal Component 2 (pca2)')
    plt.grid(False)
    plt.show()

    return df_pca["cluster"]

def get_pheno_groups(connection):
    query = "SELECT [Clone name], [Fat] FROM butterfat"
    fat_info = pd.read_sql_query(query, connection)
    fat_info.set_index(["Clone name"], inplace=True)
    fat_info = fat_info.apply(pd.to_numeric, errors='coerce')
    fat_info.dropna(inplace=True)
    fat_info.reset_index(inplace=True)
    result_df = fat_info.groupby('Clone name').mean()

    num_bins = 5
    result_df['Fat Group'] = pd.qcut(result_df['Fat'], q=num_bins, labels=False)

    sns.set(style="whitegrid")
    sns.kdeplot(data=result_df, x="Fat", hue="Fat Group", common_norm=False, fill=True, palette="viridis", linewidth=2.5)
    plt.title('Density of Fat for Each Group')
    plt.xlabel('Fat')
    plt.ylabel('Density')
    plt.legend(title='Fat Group', loc='upper right', labels=[f'Group {i}' for i in range(result_df['Fat Group'].nunique())])
    plt.grid(False)
    plt.show()

    return result_df['Fat Group']


def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    
    geo_groups = get_geo_groups(connection)
    geno_groups = get_geno_groups(connection)
    pheno_groups = get_pheno_groups(connection)

    merged_result = pd.merge(geo_groups, geno_groups, left_index=True, right_index=True, how='inner')
    merged_result = pd.merge(merged_result, pheno_groups, left_index=True, right_index=True, how='inner')

    connection.close()

if __name__ == "__main__":
    main()

