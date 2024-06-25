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

def main():
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
    for name, df in all_dfs.items():
        df_dummies = dummies_order(df)
        all_dfs[name] = df_dummies
    
main()