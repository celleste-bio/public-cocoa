"""
connection between:
* geo group -> continent
* geno group -> kmean on ssr validated visualy with pca
* pheno group -> fat precentage binned
* compere between geno group and ssr
"""


# packages
import os
import sqlite3 as sqlite
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns


def go_back_dir(path, number):
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def fillna_mode(group):
    return group.fillna(value=group.mode().iloc[0])

def create_path(script_path):
    # context
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    return(db_path)



def combine_groups_ssr(connection):
   
    # Fetch data from the 'groups' table
    groups_query = "SELECT [Clone name], [Refcode], [Population] FROM groups"
    groups_info = pd.read_sql_query(groups_query, connection)
    groups_info.dropna(inplace=True)
    groups_info.drop_duplicates(inplace=True)
    label_encoder = LabelEncoder()
    groups_info['Population_encoded'] = label_encoder.fit_transform(groups_info['Population'])

    # Fetch data from the 'ssr' table
    ssr_query = "SELECT * FROM ssr"
    ssr = pd.read_sql_query(ssr_query, connection)

    merged_data = pd.merge(groups_info, ssr, on=['Clone name', 'Refcode'])


    # merged_data.set_index(["Clone name"], inplace=True)
    # numeric_merged_data = merged_data.apply(pd.to_numeric, errors='coerce')
    # numeric_merged_data = numeric_merged_data[numeric_merged_data['Clone name'] != '-']
    # numeric_merged_data.reset_index(inplace=True)
    # groups_ssr_info = numeric_merged_data.groupby(["Clone name"])
    # merged_filled = groups_ssr_info.apply(fillna_mode)
    # merged_result = merged_filled.reset_index(drop=True).apply(fillna_mode)

    # k_values = range(2, 11)
    # silhouette_scores = []
    # for k in k_values:
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     cluster_labels = kmeans.fit_predict(merged_result)
    #     silhouette_scores.append(silhouette_score(merged_result, cluster_labels))

    # plt.plot(k_values, silhouette_scores, marker='o')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Analysis for Optimal k')
    # plt.show()

    # silhouette_scores
    # min_score_index = silhouette_scores.index(min(silhouette_scores))
    # optimal_k = min_score_index + k_values[0]

    # key_columns = ['Clone name', 'Refcode']

    # # Select features for clustering (excluding the key columns)
    # features = ssr.drop(key_columns, axis=1)

    # # Perform KMeans clustering
    # kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust number of clusters as needed
    # ssr['Cluster'] = kmeans.fit_predict(features)

    # # Step 2: Merge cluster labels with the 'groups' table
    # groups_query = "SELECT [Clone name], [Refcode], [Population] FROM groups"
    # groups = pd.read_sql_query(groups_query, connection)

    # # Merge on key columns
    # merged_data = pd.merge(groups, ssr[key_columns + ['Cluster']], on=key_columns)

    # # Step 3: Analyze the relationship between clusters and 'Population'
    # cluster_population_relationship = merged_data.groupby('Cluster')['Population'].value_counts(normalize=True).unstack()

    # kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    # cluster_labels = kmeans.fit_predict(merged_result)

    # pca = PCA(n_components=2)
    # df_pca = pd.DataFrame(pca.fit_transform(merged_result), columns=["pca1", "pca2"])
    # df_pca.index = merged_result.index
    # df_pca["cluster"] = cluster_labels

    # plt.scatter(df_pca['pca1'], df_pca['pca2'], c=df_pca["cluster"], alpha=0.7, edgecolors='w', linewidth=0.5)
    # plt.title('PCA Scatter Plot')
    # plt.xlabel('Principal Component 1 (pca1)')
    # plt.ylabel('Principal Component 2 (pca2)')
    # plt.grid(False)
    # plt.show()






def get_geo_pheno_groups(connection):
    query = "SELECT [Refcode], [Clone name], [Country] FROM planting"
    planting_info = pd.read_sql_query(query, connection)
    planting_info.dropna(inplace=True)
    planting_info = planting_info[planting_info['Clone name'] != '-']
    planting_info.drop_duplicates(inplace=True)
    planting_info.set_index(["Refcode", "Clone name"], inplace=True)
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


def get_geno_ssr(connection):
    query = "SELECT * FROM ssr"
    ssr_info = pd.read_sql_query(query, connection)
    ssr_info.drop(columns=["Refcode"], inplace=True)
    ssr_info.set_index(["Clone name"], inplace=True)
    numeric_ssr_info = ssr_info.apply(pd.to_numeric, errors='coerce')
    numeric_ssr_info = numeric_ssr_info[numeric_ssr_info['Clone name'] != '-']
    numeric_ssr_info.reset_index(inplace=True)
    grouped_ssr_info = numeric_ssr_info.groupby(["Clone name"])
    ssr_filled = grouped_ssr_info.apply(fillna_mode)
    ssr_result = ssr_filled.reset_index(drop=True).apply(fillna_mode)
    ssr_result.set_index(["Clone name"], inplace=True)


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
    plt.show()

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

def get_butterfatTable(connection):
    query="SELECT [Clone name], [Refcode], [Fat] FROM butterfat"
    butterfat_info = pd.read_sql_query(query, connection)
    butterfat_info.drop(columns=["Refcode"], inplace=True)
    butterfat_info.set_index(["Clone name"], inplace=True)



def main():
    #create database
    script_path = os.path.realpath(__file__)
    db_path=create_path(script_path)
    #Create a SQLite database connection
    connection = sqlite.connect(db_path)

    connection.close()

if __name__ == "__main__":
    main()



