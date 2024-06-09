"""
clean ICGD data
"""

# packages
import os
import sys
import sqlite3 as sqlite
import pandas as pd
import numpy as np
import yaml

sys.path.append("/home/public-cocoa/src/")
from path_utils import go_back_dir

def get_configs(script_path):
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_cleanning.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def replace_dash_with_nan(df):
    # Replace "-" with NaN in both numerical and categorical columns
    df = df.replace('-', np.nan)
    return df


def filter_high_missing_columns(df, threshold):
    """
    Filters out columns with more than threshold % missing values
    """
    missing_percent = (df.isnull().mean() * 100).round(2)
    columns_to_drop = missing_percent[missing_percent > threshold].index
    df = df.drop(columns=columns_to_drop)
    return df

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
    # script_path="/home/public-cocoa/src/eda/clean_data.py"
    script_path = os.path.realpath(__file__)
    config = get_configs(script_path)
    connection = connection = sqlite.connect(config["db_path"])

    # create dataFrame
    df = create_data_frame(connection, config["tables"], config["id_columns"])
    df = replace_dash_with_nan(df)
    df = filter_high_missing_columns(df, config["missing_threshold"])

    target_column = [col for col in df.columns if col_name_template(config["target_column"]) in col][0]
    df = df.dropna(subset=[target_column])

    # df.info()
    connection.close()

    project_path = go_back_dir(script_path, 1)
    output_path = os.path.join(project_path, "data", "cleaned_data.csv")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()