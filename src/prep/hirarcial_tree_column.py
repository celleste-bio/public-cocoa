"""
add hirarcial tree column
"""


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

# Read the CSV files
df = pd.read_csv('/content/merged_full_data_with_full_names.csv')

df.info()


# Assuming your dataframe is named 'df'
# Select the numerical columns for clustering
X = df.select_dtypes(include=['float64']).values

# Instantiate the AgglomerativeClustering object with 2 clusters
clustering = AgglomerativeClustering(n_clusters=2)

# Fit the clustering algorithm to the data
cluster_labels = clustering.fit_predict(X)

# Add a new column 'cluster' to the dataframe
df['cluster'] = cluster_labels


# Hierarchical Clustering
plt.figure(figsize=(10, 7))
dend = shc.dendrogram(shc.linkage(df, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()


from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Perform hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = clustering.fit_predict(df)
# Calculate silhouette score
silhouette_avg = silhouette_score(data, labels)
print("Silhouette Score:", silhouette_avg)
davies_bouldin_index = davies_bouldin_score(data, labels)
print("Davies-Bouldin Index:", davies_bouldin_index)
# Calinski-Harabasz Index
ch_index = calinski_harabasz_score(data, labels)
print("Calinski-Harabasz Index:", ch_index)
