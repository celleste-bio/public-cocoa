# packages
import os
import sys
import sqlite3 as sqlite
import pandas as pd
# import numpy as np

sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir
from utils import read_yaml

sys.path.append("/home/public-cocoa/src/prep")
from clean_data import clean_data
from dummies_order import dummies_order
from fill_missing_values import fill_missing_values
from Kmeans_missing_value import clustering_and_replace_missing_values
from Median_and_Frequent_Function import fill_missing_values_median_and_mode
from Normal_and_Frequent_Function import Normal_and_Frequent_Function
from outliers_by_IQR import IQR_outliers
from hirarcial_tree_column import hirrarcial_tree


def add_table_prefix(df, table_name, id_columns):
    """
    Adds table name prefix to non id columns
    """
    prefixed_columns = {}
    for col in df.columns:
        if col not in id_columns:
            prefixed_columns[col] = f"{table_name}_{col}"
        else:
            prefixed_columns[col] = col
    return df.rename(columns=prefixed_columns)

def col_name_template(col_name):
    return col_name.replace(' ', '_').lower()

def create_data_frame(conn, tables, id_columns):
    """
    Aggregates and preppering tables
    """
    merged_df = pd.DataFrame(columns=id_columns)
    for table in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df = add_table_prefix(df, table, id_columns)
        merged_df = pd.merge(df, merged_df, on=id_columns, how='left')

    merged_df = merged_df.drop_duplicates(subset=id_columns)
    merged_df = merged_df.rename(columns=lambda x: col_name_template(x))
    return merged_df

def drop_columns_from_df(df, columns_to_drop):
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df

def prep_data(data,method,outliers,Hierarchical_tree_clusters,C_OR_WC):
    # script_path="/home/public-cocoa/src/prep/main.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_main.yaml")
    config = read_yaml(config_path)
    cleaned_data = data.copy()
    cleaned_data=clean_data(data)


    if method == "KM":
        filled_df = clustering_and_replace_missing_values(cleaned_data, 3)
    elif method == "ME":
            filled_df = fill_missing_values_median_and_mode(cleaned_data)
    elif method == "NO":
            filled_df = Normal_and_Frequent_Function(cleaned_data)
    
    if C_OR_WC.lower() in ['wc']:
        filled_df = drop_columns_from_df(filled_df, config["columns_cotyledon"])

    if outliers==True:
         filled_df = IQR_outliers(filled_df)

    df_dummies = dummies_order(filled_df)
    df_dummies.columns = df_dummies.columns.str.replace('<=', 'lte', regex=False)
    df_dummies.columns = df_dummies.columns.str.replace('>', 'gt', regex=False)

    df_dummies = hirrarcial_tree(df_dummies, Hierarchical_tree_clusters)

    return df_dummies






def main_prep():
    # script_path="/home/public-cocoa/src/prep/main.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_main.yaml")
    config = read_yaml(config_path)

    connection = sqlite.connect(config["db_path"])
    data = create_data_frame(connection, config["tables"], config["id_columns"])
    cleaned_data=clean_data(data)
    #the data is clean without duplication and without null in target column

    all_dfs=fill_missing_values(cleaned_data)
#    for name, df in all_dfs.items():
#       df_dummies = dummies_order(df)
#       all_dfs[name] = df_dummies
    
    # Create the output directory if it doesn't exist
    output_dir = config["save_praped_data"]
    os.makedirs(output_dir, exist_ok=True)

    # Write each DataFrame to a separate CSV file
    for name, df in all_dfs.items():
        csv_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        
if __name__ == "__main__":
    main_prep()