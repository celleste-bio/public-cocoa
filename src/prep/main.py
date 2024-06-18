# packages
import os
import sys
import sqlite3 as sqlite
import pandas as pd
import numpy as np
import yaml

sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir
from clean_data import clean_data
from Kmeans_missing_value import clustering_and_replace_missing_values
from Median_and_Frequent_Function import fill_missing_values_median_and_mode
from Normal_and_Frequent_Function import fill_missing_values_Normal

def get_configs(script_path):
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_cleanning.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

def main():
    # script_path="/home/public-cocoa/src/prep/main.py"
    script_path = os.path.realpath(__file__)
    config = get_configs(script_path)
    connection = sqlite.connect(config["db_path"])
    df = create_data_frame(connection, config["tables"], config["id_columns"])
    data = df.copy()
    data = clean_data(data,config)
    #the data is clean without duplication and without null in target column

    # Splitting the DataFrame into train and test sets
    train_df_75 = data.sample(frac=0.75, random_state=42)  # 75% for training
    test_df_25 = data.drop(train_df_75.index)

    # Splitting the DataFrame into train and test sets
    train_df_80 = data.sample(frac=0.80, random_state=42)  # 80% for training
    test_df_20 = data.drop(train_df_80.index)
    train_df_80.info()


    #filling missing values
    id_columns = config['id_columns_new_name']
    df_80_KM=clustering_and_replace_missing_values(train_df_80,3,id_columns)
    df_75_KM=clustering_and_replace_missing_values(train_df_75,3,id_columns)
    df_80_ME=fill_missing_values_median_and_mode(train_df_80)
    df_75_ME=fill_missing_values_median_and_mode(train_df_75)
    df_80_NO=