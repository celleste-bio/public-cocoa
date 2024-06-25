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

sys.path.append("/home/public-cocoa/src/prep/")
from path_utils import go_back_dir

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

def convert_to_float(df):
        # Identify object columns that might be numeric
    object_cols = df.select_dtypes(include=['object']).columns
    numeric_cols = []

    # Attempt to convert to float and see which columns succeed
    for col in object_cols:
        try:
            df[col] = df[col].astype(float)
            numeric_cols.append(col)
        except ValueError:
            continue

    print("Converted columns to float:", numeric_cols)

    # Check the data types of the DataFrame after conversion
    print(df.dtypes)
    return df

def col_name_template(col_name):
    return col_name.replace(' ', '_').lower()

def drop_columns_from_df(df, columns_to_drop):
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df

def clean_data_function(df_clean, config):
    # script_path="/home/public-cocoa/src/prep/clean_data.py"
    # script_path = os.path.realpath(__file__)
    # config = get_configs(config_script_path)

    df_clean = replace_dash_with_nan(df_clean)
    df_clean = filter_high_missing_columns(df_clean, config["missing_threshold"])

    target_column = [col for col in df_clean.columns if col_name_template(config["target_column"]) in col][0]
    df_clean = df_clean.dropna(subset=[target_column])
    id_columns = config['id_columns_new_name']
    df_clean.info()
    df_clean = drop_columns_from_df(df_clean, id_columns)
    df_clean = drop_columns_from_df(df_clean,target_column)
    df_clean=convert_to_float(df_clean)
    return df_clean
    # df.info()
    #connection.close()

    #project_path = go_back_dir(script_path, 2)
    #output_path = os.path.join(project_path, "data", "cleaned_data.csv")
    #df.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_data_function()



