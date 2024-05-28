"""
connection between:
* geo group -> continent
* geno group -> kmean on ssr validated visualy with pca
* pheno group -> fat precentage binned
"""

# packages
import os
import sys
import sqlite3 as sqlite
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/home/public-cocoa/src/")
from path_utils import go_back_dir

def fillna_mode(group):
    return group.fillna(value=group.mode().iloc[0])
    
def get_geo_groups(connection):
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

    sns.set(style="whitegrid")
    sns.kdeplot(data=result_df, x="Fat", common_norm=False, fill=True, palette="viridis", linewidth=2.5)
    plt.title('Density of Fat percentage')
    plt.xlabel('Fat')
    plt.ylabel('Density')
    plt.grid(False)
    plt.show()

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

    query = "SELECT [Refcode], [Clone name], [Fat] FROM butterfat"
    fat_info = pd.read_sql_query(query, connection)
    fat_info.set_index(["Refcode", "Clone name"], inplace=True)
    fat_info = fat_info.apply(pd.to_numeric, errors='coerce')
    fat_info.dropna(inplace=True)
    fat_info.reset_index(inplace=True)
    fat_info = fat_info.groupby(["Refcode", "Clone name"]).mean()

    merged_result = pd.merge(fat_info, planting_info, left_index=True, right_index=True, how='inner')
    
    ref_origin = pd.read_csv(os.path.join(project_path, "ref_origin.csv"))
    ref_origin.set_index(["Refcode"], inplace=True)
    fat_info = fat_info.reset_index().set_index(["Refcode"])

    query = "SELECT refcode as Refcode, Year FROM ref_info"
    ref_info = pd.read_sql_query(query, connection)
    ref_info.set_index(["Refcode"], inplace=True)
    merged_result = pd.merge(fat_info, ref_origin, left_index=True, right_index=True, how='inner')
    merged_result = pd.merge(merged_result, ref_info, left_index=True, right_index=True, how='inner')

    # grouped_data = merged_result.groupby(['year', 'Continent'])['Fat'].agg(['mean', 'std']).unstack()
    grouped_data = merged_result.groupby(['year', 'Continent'])['Fat'].agg(['mean', 'std'])

    # Plotting a line for each continent
    for continent in merged_result['Continent'].unique():
        plt.plot(grouped_data.index, grouped_data[('mean', continent)], label=f'{continent}')

        plt.fill_between(
            grouped_data.index,
            grouped_data[('mean', continent)] - grouped_data[('std', continent)],
            grouped_data[('mean', continent)] + grouped_data[('std', continent)],
            alpha=0.3
        )

    plt.xlabel('Year')
    plt.ylabel('Fat Percentage')
    plt.title('Mean Fat Percentage Over Years with Standard Deviation for Each Continent')
    plt.legend()
    plt.grid(False)
    plt.show()
    
    return merged_result

def get_geo_ref(connection):
    # 'planting'.'Clone name'
    query = "SELECT Refcode, Country FROM planting"
    planting_info = pd.read_sql_query(query, connection)
    planting_info.drop_duplicates(inplace=True)
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
    planting_info["Continent"] = [country_continent_dict[country] for country in planting_info["Country"]]
    planting_info.set_index(["Refcode"], inplace=True)

    query = "SELECT refcode as Refcode, title as Title FROM ref_info"
    ref_info = pd.read_sql_query(query, connection)
    ref_info.set_index(["Refcode"], inplace=True)
    merged_result = pd.merge(ref_info, planting_info, left_index=True, right_index=True, how='left')


def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    
    # geo_groups = get_geo_groups(connection)
    pheno_groups = get_pheno_groups(connection)
    geno_groups = get_geno_groups(connection)
    geo_pheno_groups = get_geo_pheno_groups(connection)
    merged_df = geo_pheno_groups.reset_index().merge(geno_groups.reset_index(), on='Clone name')
    
    # merged_result = pd.merge(geo_pheno_groups, geno_groups, left_index=True, right_index=True, how='inner')

    connection.close()

if __name__ == "__main__":
    main()

from scipy.stats import f_oneway

def others():
    merged_result = pd.merge(fat_info, geno_groups, left_index=True, right_index=True, how='inner')
    sns.set(style="whitegrid")
    sns.kdeplot(data=merged_result, x="Fat", hue="cluster", common_norm=False, fill=True, palette="viridis", linewidth=2.5)
    plt.title('Density of fat percentage for each genetic group')
    plt.xlabel('Fat')
    plt.ylabel('Density')
    plt.legend(title='Genetic Group', loc='upper right', labels=[f'{i}' for i in range(merged_result['cluster'].nunique())])
    plt.grid(False)
    plt.show()

    merged_result = pd.merge(fat_info, planting_info, left_index=True, right_index=True, how='inner')
    sns.set(style="whitegrid")
    sns.kdeplot(data=merged_result, x="Fat", hue="Continent", common_norm=False, fill=True, palette="viridis", linewidth=2.5)
    plt.title('Density of fat percentage for each continent')
    plt.xlabel('Fat')
    plt.ylabel('Density')
    plt.legend(title='Continent', loc='upper right', labels=[c for c in merged_result['Continent'].unique()])
    plt.grid(False)
    plt.show()


    # Extract fat values for each cluster
    cluster_0_fat = merged_result[merged_result['cluster'] == 0]['Fat']
    cluster_1_fat = merged_result[merged_result['cluster'] == 1]['Fat']
    cluster_2_fat = merged_result[merged_result['cluster'] == 2]['Fat']
    cluster_3_fat = merged_result[merged_result['cluster'] == 3]['Fat']
    cluster_4_fat = merged_result[merged_result['cluster'] == 4]['Fat']
    cluster_5_fat = merged_result[merged_result['cluster'] == 5]['Fat']

    # Perform ANOVA test
    anova_result = f_oneway(cluster_0_fat, cluster_1_fat, cluster_2_fat, cluster_3_fat, cluster_4_fat, cluster_5_fat)

    # Print the result
    print("ANOVA p-value:", anova_result.pvalue)

    # Check if the p-value is below your significance level (e.g., 0.05) to determine significance
    if anova_result.pvalue < 0.05:
        print("There is a significant difference in mean fat between the groups.")
    else:
        print("There is no significant difference in mean fat between the groups.")

    # Extract fat values for each cluster
    asia_fat = merged_result[merged_result['Continent'] == "Asia"]['Fat']
    south_america_fat = merged_result[merged_result['Continent'] == "South America"]['Fat']
    north_america_fat = merged_result[merged_result['Continent'] == "North America"]['Fat']

    # Perform ANOVA test
    anova_result = f_oneway(asia_fat, south_america_fat, north_america_fat)

    # Print the result
    print("ANOVA p-value:", anova_result.pvalue)

    # Check if the p-value is below your significance level (e.g., 0.05) to determine significance
    if anova_result.pvalue < 0.05:
        print("There is a significant difference in mean fat between the groups.")
    else:
        print("There is no significant difference in mean fat between the groups.")

    merged_result.reset_index(inplace=True)
    merged_result['year'] = merged_result['year'].dropna().astype(int)
    sns.set(style="whitegrid")
    ax = sns.boxplot(data=merged_result, x='year', y='Fat', hue='Continent', palette='viridis')
    plt.title('Box plot of Fat by Year and Continent')
    plt.xlabel('Year')
    plt.ylabel('Fat Precentage')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.legend(title='Continent', loc='upper right')
    plt.grid(False)
    plt.show()
