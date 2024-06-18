"""
add hirarcial tree column
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def hirrarcial_tree(df,cluster_num):
    data = df.copy()
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(n_clusters=cluster_num, linkage='ward')
    labels = clustering.fit_predict(data)
    # Calculate silhouette score
    silhouette_avg = silhouette_score(data, labels)
    print("Silhouette Score:", silhouette_avg)
    davies_bouldin_index = davies_bouldin_score(data, labels)
    print("Davies-Bouldin Index:", davies_bouldin_index)
    # Calinski-Harabasz Index
    ch_index = calinski_harabasz_score(data, labels)
    print("Calinski-Harabasz Index:", ch_index)
    # Add a new column 'cluster' to the dataframe
    data['cluster'] = labels  
    return data

