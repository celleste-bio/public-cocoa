
"""
eda about flower.csv 
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
from scipy.stats import norm


from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns
from ..path_utils import go_back_dir

from sklearn.preprocessing import StandardScaler

def create_path(script_path):
    # context
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    return(db_path)

def flower(connection):
        query = "SELECT * FROM flower"
        flower_info = pd.read_sql_query(query, connection)
        query_numerical = "SELECT [Clone name], [Refcode], [Guide line length], [Sepal width], [Sepal length], [Sepal length width], [Ligule length], [Ligule width], [Ligule length width], [Ribbon length], [Ovary width], [Ovary length], [Ovary length width], [Ovule number], [Style length], [Staminode length], [Pedicel length], [Petal length] FROM flower"
        flower_numerical_data = pd.read_sql_query(query_numerical,connection)
        query_numerical_flower= "SELECT [Clone name], [Refcode], [Guide line length], [Sepal width], [Sepal length], [Sepal length width], [Ligule length], [Ligule width], [Ligule length width], [Ribbon length] FROM flower"
        flower_numerical_data_split = pd.read_sql_query(query_numerical_flower,connection)
        flower_numerical_data_split = flower_numerical_data_split.replace('-', float('nan'))
        flower_numerical_data_split.info()
        flower_numerical_data.info()
        flower_numerical_data = flower_numerical_data.replace('-', float('nan'))
        flower_numerical_data = flower_numerical_data.fillna()
        # Get the list of all column names except 'Clone name' and 'Refcode'
        columns_to_convert = [col for col in flower_numerical_data.columns if col not in ['Clone name', 'Refcode']]

        # Convert selected columns to float
        flower_numerical_data[columns_to_convert] = flower_numerical_data[columns_to_convert].apply(pd.to_numeric, errors='coerce')
        flower_numerical_data.info()
        # Iterate over columns
        for column in flower_numerical_data.columns:
        # Check if column contains float values
            if flower_numerical_data[column].dtype == 'float64':
              # Plot histogram
                plt.figure(figsize=(8, 6))
                sns.histplot(flower_numerical_data[column].dropna(), bins=10, color='skyblue', edgecolor='black',kde=True,stat="density")
                plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.grid(True)
                
                #plt.show()

                mu, std = norm.fit(flower_numerical_data[column].dropna())
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, 'k', linewidth=2)
                plt.show()


# def pipline_fun(connection):
#     query_numerical_flower= "SELECT [Clone name], [Refcode], [Guide line length], [Sepal width], [Sepal length], [Sepal length width], [Ligule length], [Ligule width], [Ligule length width], [Ribbon length] FROM flower"
#     flower_numerical_data_split = pd.read_sql_query(query_numerical_flower,connection)
#     flower_numerical_data.info()
#     flower_numerical_data = flower_numerical_data.replace('-', float('nan'))

#     # Define a pipeline for preprocessing and classification
#     pipeline = Pipeline([
#         ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),  # Impute missing values
#         ('classifier', KNeighborsClassifier())  # KNN classifier
#         ])

#     # Fit the pipeline to your data
#     pipeline.fit(X_train, y_train)

#     # Predict using the fitted pipeline
#     predictions = pipeline.predict(X_test)


def Kmeans_func(df):
    # Concatenate 'Clone name' and 'Refcode' columns with a separator
    index = df['Clone name'] + '_' + df['Refcode']

    # Set the concatenated series as the index of the DataFrame
    df.set_index(index, inplace=True)

    # Drop the original 'Clone name' and 'Refcode' columns if needed
    df.drop(columns=['Clone name', 'Refcode'], inplace=True)
    df.info()
    # Drop non-numeric columns and rows with NaN values
    #df = df.select_dtypes(include=['float64']).dropna()

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Create KMeans model
    kmeans = KMeans(n_clusters=3)  # Adjust number of clusters as needed
    kmeans.fit(scaled_data)

    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Print the number of samples in each cluster
    print(df['Cluster'].value_counts())

    # Visualize the clusters

    for cluster in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data['Guide line length'], cluster_data['Cluster'], label=f'Cluster {cluster}')
        plt.xlabel('Guide line length')
        plt.ylabel('Cluster')
        plt.title('Clusters')
        plt.legend()
        plt.show()



       

def main():
    #script_path="/home/public-cocoa/src/eda/flower_eda.py"
    #create database
    script_path = os.path.realpath(__file__)
    db_path=create_path(script_path)
    #Create a SQLite database connection
    connection = sqlite.connect(db_path)

    connection.close()

if __name__ == "__main__":
    main()
