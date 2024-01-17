"""
exploring beans data
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    bean_info = pd.read_sql_query("SELECT * FROM bean", connection)
    imputer = SimpleImputer(strategy='mean')
    to_keep = ['Refcode', 'Clone name', 'Number', 'Number (max)', 'Length', 'Width',
        'Total dry weight', 'Wet weight', 'Dry weight','Total wet weight','Cotyledon dry weight',
        'Cotyledon length', 'Cotyledon width', 'Cotyledon wet weight', 'Cotyledon thickness', 'Thickness']
    bean_info_num = bean_info[to_keep].set_index(['Refcode', 'Clone name'])

    for col in bean_info_num.columns:
        bean_info_num[col] = pd.to_numeric(bean_info_num[col], errors='coerce')

    bean_info_num = pd.DataFrame(imputer.fit_transform(bean_info_num), columns=bean_info_num.columns, index=bean_info_num.index)

    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(bean_info_num), columns=["pca1", "pca2"])

    plt.scatter(df_pca['pca1'], df_pca['pca2'], alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.title('PCA Scatter Plot')
    plt.xlabel('Principal Component 1 (pca1)')
    plt.ylabel('Principal Component 2 (pca2)')
    plt.grid(False)
    plt.show()

    connection.close()

if __name__ == "__main__":
    main()

