import os
import sqlite3 as sqlite
import pandas as pd
import yaml

sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir

sys.path.append("/home/public-cocoa/src/prep")
from clean_data import clean_data_function
from Kmeans_missing_value import clustering_and_replace_missing_values
from Median_and_Frequent_Function import fill_missing_values_median_and_mode
from Normal_and_Frequent_Function import fill_missing_values
from outliers_by_IQR import IQR_outliers
from hirarcial_tree_column import hirrarcial_tree
from dummies_order import dummies_order_func

def get_configs(script_path):
    config_path = os.path.join(os.path.dirname(script_path), 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_data_frame(connection, tables, id_columns):
    query = f"SELECT {', '.join(id_columns)} FROM {', '.join(tables)}"
    return pd.read_sql_query(query, connection)

def clean_data_function(data, config):
    data=clean_data_function(data,config)
    return data

def split_data(data, frac, random_state):
    train_df = data.sample(frac=frac, random_state=random_state)
    test_df = data.drop(train_df.index)
    return train_df, test_df

def handle_outliers(df):
    df=IQR_outliers(df)
    return df

def fill_missing_values_method(df, method):
    if method == 'KM':
        df = clustering_and_replace_missing_values(df,k=3)
    if method == 'ME':
        df = fill_missing_values_median_and_mode(df)
    if method == 'NO':
        df = fill_missing_values(df)
    return df

def dummies_order_func(df, config):
    # Implement your dummy variable creation logic here
    return df

def hirrarcial_tree(df, num_clusters):
    # Implement your hierarchical tree logic here
    return df

# Main script
def main():
    #script_path="/home/public-cocoa/src/prep/main_new.py"
    script_path = os.path.realpath(__file__)
    config = get_configs(script_path)

    connection = sqlite.connect(config["db_path"])
    df = create_data_frame(connection, config["tables"], config["id_columns"])
    df.info()

    data = df.copy()
    data.info()

    cleaned_data = clean_data_function(data, config)
    cleaned_data.info()

    # Splitting the DataFrame into train and test sets
    train_df_75, test_df_25 = split_data(cleaned_data, 0.75, 42)
    train_df_80, test_df_20 = split_data(cleaned_data, 0.80, 42)

    # Handling outliers
    df_75_O = handle_outliers(train_df_75)
    df_75_WO = train_df_75.copy()
    df_80_O = handle_outliers(train_df_80)
    df_80_WO = train_df_80.copy()

    # Filling missing values
    methods = ['KM', 'ME', 'NO']
    dfs_75 = {'O': df_75_O, 'WO': df_75_WO}
    dfs_80 = {'O': df_80_O, 'WO': df_80_WO}

    for method in methods:
        for key in dfs_75:
            dfs_75[key + '_' + method] = fill_missing_values_method(dfs_75[key], method)
        for key in dfs_80:
            dfs_80[key + '_' + method] = fill_missing_values_method(dfs_80[key], method)

    # Creating dummies & ordinary
    for method in methods:
        for key in dfs_75:
            dfs_75[key + '_' + method] = dummies_order_func(dfs_75[key + '_' + method], config)
        for key in dfs_80:
            dfs_80[key + '_' + method] = dummies_order_func(dfs_80[key + '_' + method], config)

    # Hierarchical tree
    clusters = [0, 2, 4, 8]
    for method in methods:
        for key in dfs_75:
            for cluster in clusters:
                dfs_75[key + '_' + method + '_' + str(cluster)] = hirrarcial_tree(dfs_75[key + '_' + method], cluster)
        for key in dfs_80:
            for cluster in clusters:
                dfs_80[key + '_' + method + '_' + str(cluster)] = hirrarcial_tree(dfs_80[key + '_' + method], cluster)