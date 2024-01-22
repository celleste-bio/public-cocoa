"""
exploring ssr data
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

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
    numeric_ssr_info.replace({np.nan: 1e10}, inplace=True)
    numeric_ssr_info['Clone name'] = ssr_info['Clone name']
    grouped_ssr_info = numeric_ssr_info.groupby('Clone name').mean()
    grouped_ssr_info.drop(columns=["Refcode"], inplace=True)
    distances = pdist(grouped_ssr_info, metric='euclidean')
    distance_matrix = squareform(distances)
    linkage_matrix = linkage(distances, method='average')



if __name__ == "__main__":
    main()

