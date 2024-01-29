"""
exploring ssr data and pynotype connections
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def go_back_dir(path, number):
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def fillna_mode(group):
    return group.fillna(value=group.mode().iloc[0])
    
def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    
    # markers data
    query = """
            SELECT
            "ssr".*,
            "butterfat"."Fat"
        FROM
            "ssr"
        JOIN
            "butterfat" ON "ssr"."Clone name" = "butterfat"."Clone name"
    """

    data = pd.read_sql_query(query, connection)
    data.drop(columns=["Refcode"], inplace=True)
    data.set_index(["Clone name"], inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data.reset_index(inplace=True)
    grouped_data = data.groupby(['Clone name'])
    data_filled = grouped_data.apply(fillna_mode).reset_index(drop=True)
    result_df = data_filled.dropna().drop_duplicates().groupby('Clone name').mean()

    # Calculate distances using pdist
    distances = pdist(result_df, metric='euclidean')
    distance_matrix = squareform(distances)
    linkage_matrix = linkage(distances, method='average')

    tree = dendrogram(linkage_matrix, labels=result_df.index, orientation='top', distance_sort='ascending')
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Clone name')
    plt.ylabel('Distance')
    plt.show()

    order = tree['leaves']
    result_df_ordered = result_df.iloc[order]
    sns.heatmap(result_df_ordered[["Fat"]].T, cmap='viridis', cbar_kws={'label': 'Fat'}, annot=True)

    #################################################################################################
     
    num_bins = 5
    result_df['Fat_Category'] = pd.cut(result_df['Fat'], bins=num_bins, labels=False)

    markers_df = result_df.drop(columns=["Fat",'Fat_Category'])
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(markers_df), columns=["pca1", "pca2"])
    df_pca.index = markers_df.index
    merged_df = pd.merge(df_pca, result_df['Fat_Category'], left_index=True, right_index=True)

    plt.scatter(merged_df['pca1'], merged_df['pca2'], c=merged_df['Fat_Category'], cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
    cbar = plt.colorbar()
    cbar.set_label('Fat')
    plt.title('PCA Scatter Plot - Colored by Fat')
    plt.xlabel('Principal Component 1 (pca1)')
    plt.ylabel('Principal Component 2 (pca2)')
    plt.show()

    #################################################################################################

    result_df.sort_values("Fat", inplace=True)
    markers_df = result_df.drop(columns=["Fat"])

    # Create t-SNE model
    tsne = TSNE(n_components=2, random_state=42)

    # Fit and transform the data
    markers_tsne = tsne.fit_transform(markers_df)

    # Combine the t-SNE results with the target variable (Fat)
    markers_tsne_df = pd.DataFrame(markers_tsne, columns=['tsne1', 'tsne2'])
    markers_tsne_df.index = result_df.index
    markers_tsne_df['Fat'] = result_df["Fat"]

    plt.scatter(markers_tsne_df['tsne1'], markers_tsne_df['tsne2'], c=markers_tsne_df['Fat'], cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
    cbar = plt.colorbar()
    cbar.set_label('Fat')
    plt.title('PCA Scatter Plot - Colored by Fat')
    plt.xlabel('Principal Component 1 (tsne1)')
    plt.ylabel('Principal Component 2 (tsne2)')
    plt.show()

#################################################################################################
    
    num_bins = 2

    result_df['Fat_Category'] = pd.cut(result_df['Fat'], bins=num_bins, labels=False)

    target_variable = result_df['Fat_Category']

    markers_df = result_df.drop(columns=["Fat",'Fat_Category'])

    lda = LinearDiscriminantAnalysis(n_components=2)

    markers_lda = lda.fit_transform(markers_df, target_variable)

    markers_lda_df = pd.DataFrame(markers_lda, columns=['lda1', 'lda2'])
    markers_lda_df.index = result_df.index
    markers_lda_df['Fat_Category'] = result_df['Fat_Category']

    plt.scatter(markers_lda_df['lda1'], markers_lda_df['lda2'], c=markers_lda_df['Fat_Category'], cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
    cbar = plt.colorbar()
    cbar.set_label('Fat')
    plt.title('LDA Scatter Plot - Colored by Fat')
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.show()



    connection.close()

if __name__ == "__main__":
    main()

