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

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    # Load your data
    data = merged_data  # Replace "your_data.csv" with the path to your dataset
    # Replace '-' with 0
    data.replace('-', 0, inplace=True)
    # Impute missing values with the mean
    # Drop rows with missing values
    data.dropna(inplace=True)


    # Extract features and target variable
    X = data[['M1 a1', 'M1 a2', 'M6 a1', 'M6 a2', 'M8 a1', 'M8 a2', 'M12 a1', 'M12 a2', 'M15 a1', 'M15 a2', 'M18 a1', 'M18 a2', 'M22 a1', 'M22 a2', 'M24 a1', 'M24 a2', 'M26 a1', 'M26 a2', 'M33 a1', 'M33 a2', 'M37 a1', 'M37 a2', 'M60 a1', 'M60 a2']]
    y = data["Population_encoded"]  # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Number of rows in the dataset:", len(data))

    # Create and train the decision tree model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Predict using the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    

    # Optionally, visualize the decision tree
    plt.figure(figsize=(20, 10))  # Adjust the figure size as needed
    class_names = [str(c) for c in y.unique()]
    plot_tree(clf, feature_names=X.columns, class_names=class_names, filled=True)
    plt.show()



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

    
    #kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans = KMeans(n_clusters=10, random_state=42)
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
    #script_path="/home/public-cocoa/src/eda/geo_geno_pheno_dafni.py"
    #create database
    script_path = os.path.realpath(__file__)
    db_path=create_path(script_path)
    #Create a SQLite database connection
    connection = sqlite.connect(db_path)

    connection.close()

if __name__ == "__main__":
    main()



