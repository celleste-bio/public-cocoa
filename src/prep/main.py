# packages
import os
import sys
import sqlite3 as sqlite
import pandas as pd
import numpy as np
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

def drop_columns_from_df(df, columns_to_drop):
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df

def main():
    # script_path="/home/public-cocoa/src/prep/main.py"
    script_path = os.path.realpath(__file__)
    config = get_configs(script_path)
    connection = sqlite.connect(config["db_path"])
    df = create_data_frame(connection, config["tables"], config["id_columns"])
    df.info()
    data = df.copy()
    data.info()
    cleaned_data=clean_data_function(data,config)
    #the data is clean without duplication and without null in target column
    cleaned_data.info()
    # Splitting the DataFrame into train and test sets
    train_df_75 = cleaned_data.sample(frac=0.75, random_state=42)  # 75% for training
    test_df_25 = cleaned_data.drop(train_df_75.index)

    # Splitting the DataFrame into train and test sets
    train_df_80 = cleaned_data.sample(frac=0.80, random_state=42)  # 80% for training
    test_df_20 = cleaned_data.drop(train_df_80.index)
    train_df_80.info()

    df_80_O=IQR_outliers(train_df_80)
    df_80_WO=train_df_80.copy()
    df_75_O=IQR_outliers(train_df_75)
    df_75_WO=train_df_75.copy()

    # Initialize dictionary to store all DataFrames
    all_dfs = {}
    
    #filling missing values
    df_80_O_KM=clustering_and_replace_missing_values(df_80_O,3)
    df_80_WO_KM=clustering_and_replace_missing_values(df_80_WO,3)
    df_75_O_KM=clustering_and_replace_missing_values(df_75_O,3)
    df_75_WO_KM=clustering_and_replace_missing_values(df_75_WO,3)
    df_80_O_ME=fill_missing_values_median_and_mode(df_80_O)
    df_80_WO_ME=fill_missing_values_median_and_mode(df_80_WO)
    df_75_O_ME=fill_missing_values_median_and_mode(df_75_O)
    df_75_WO_ME=fill_missing_values_median_and_mode(df_75_WO)
    df_80_O_NO=fill_missing_values(df_80_O)
    df_80_WO_NO=fill_missing_values(df_80_WO)
    df_75_O_NO=fill_missing_values(df_75_O)
    df_75_WO_NO=fill_missing_values(df_75_WO)

    #dummies & ordinary
    
    df_80_O_KM=dummies_order_func(df_80_O_KM,config)
    df_80_WO_KM=dummies_order_func(df_80_WO_KM,config)
    df_75_O_KM=dummies_order_func(df_75_O_KM,config)
    df_75_WO_KM=dummies_order_func(df_75_WO_KM,config)
    df_80_O_ME=dummies_order_func(df_80_O_ME,config)
    df_80_WO_ME=dummies_order_func(df_80_WO_ME,config)
    df_75_O_ME=dummies_order_func(df_75_O_ME,config)
    df_75_WO_ME=dummies_order_func(df_75_WO_ME,config)
    df_80_O_NO=dummies_order_func(df_80_O_NO,config)
    df_80_WO_NO=dummies_order_func(df_80_WO_NO,config)
    df_75_O_NO=dummies_order_func(df_75_O_NO,config)
    df_75_WO_NO=dummies_order_func(df_75_WO_NO,config)

    #hirrarcial tree

    df_80_O_KM_0=df_80_O_KM.copy()
    df_80_O_KM_2=hirrarcial_tree(df_80_O_KM,2)
    df_80_O_KM_4=hirrarcial_tree(df_80_O_KM,4)
    df_80_O_KM_8=hirrarcial_tree(df_80_O_KM,8)
    df_80_WO_KM_0=df_80_WO_KM.copy()
    df_80_WO_KM_2=hirrarcial_tree(df_80_WO_KM,2)
    df_80_WO_KM_4=hirrarcial_tree(df_80_WO_KM,4)
    df_80_WO_KM_8=hirrarcial_tree(df_80_WO_KM,8)
    df_75_O_KM_0=df_75_O_KM.copy()
    df_75_O_KM_2=hirrarcial_tree(df_75_O_KM,2)
    df_75_O_KM_4=hirrarcial_tree(df_75_O_KM,4)
    df_75_O_KM_8=hirrarcial_tree(df_75_O_KM,8)
    df_75_WO_KM_0=df_75_WO_KM.copy()
    df_75_WO_KM_2=hirrarcial_tree(df_75_WO_KM,2)
    df_75_WO_KM_4=hirrarcial_tree(df_75_WO_KM,4)
    df_75_WO_KM_8=hirrarcial_tree(df_75_WO_KM,8)

    df_80_O_ME_0=df_80_O_ME.copy()
    df_80_O_ME_2=hirrarcial_tree(df_80_O_ME,2)
    df_80_O_ME_4=hirrarcial_tree(df_80_O_ME,4)
    df_80_O_ME_8=hirrarcial_tree(df_80_O_ME,8)
    df_80_WO_ME_0=df_80_WO_ME.copy()
    df_80_WO_ME_2=hirrarcial_tree(df_80_WO_ME,2)
    df_80_WO_ME_4=hirrarcial_tree(df_80_WO_ME,4)
    df_80_WO_ME_8=hirrarcial_tree(df_80_WO_ME,8)
    df_75_O_ME_0=df_75_O_ME.copy()
    df_75_O_ME_2=hirrarcial_tree(df_75_O_ME,2)
    df_75_O_ME_4=hirrarcial_tree(df_75_O_ME,4)
    df_75_O_ME_8=hirrarcial_tree(df_75_O_ME,8)
    df_75_WO_ME_0=df_75_WO_ME.copy()
    df_75_WO_ME_2=hirrarcial_tree(df_75_WO_ME,2)
    df_75_WO_ME_4=hirrarcial_tree(df_75_WO_ME,4)
    df_75_WO_ME_8=hirrarcial_tree(df_75_WO_ME,8)

    df_80_O_NO_0=df_80_O_NO.copy()
    df_80_O_NO_2=hirrarcial_tree(df_80_O_NO,2)
    df_80_O_NO_4=hirrarcial_tree(df_80_O_NO,4)
    df_80_O_NO_8=hirrarcial_tree(df_80_O_NO,8)
    df_80_WO_NO_0=df_80_WO_NO.copy()
    df_80_WO_NO_2=hirrarcial_tree(df_80_WO_NO,2)
    df_80_WO_NO_4=hirrarcial_tree(df_80_WO_NO,4)
    df_80_WO_NO_8=hirrarcial_tree(df_80_WO_NO,8)
    df_75_O_NO_0=df_75_O_NO.copy()
    df_75_O_NO_2=hirrarcial_tree(df_75_O_NO,2)
    df_75_O_NO_4=hirrarcial_tree(df_75_O_NO,4)
    df_75_O_NO_8=hirrarcial_tree(df_75_O_NO,8)
    df_75_WO_NO_0=df_75_WO_NO.copy()
    df_75_WO_NO_2=hirrarcial_tree(df_75_WO_NO,2)
    df_75_WO_NO_4=hirrarcial_tree(df_75_WO_NO,4)
    df_75_WO_NO_8=hirrarcial_tree(df_75_WO_NO,8)

    #drop cotyhlidon
    columns_to_drop = config
    drop_columns_from_df(df, columns_to_drop)

    #km
    df_80_O_KM_0_C=df_80_O_KM_0.copy()
    df_80_O_KM_0_WC=drop_columns_from_df(df_80_O_KM_0,columns_to_drop)
    df_80_O_KM_2=df_80_O_KM_0.copy()
    df_80_O_KM_2_WC=drop_columns_from_df(df_80_O_KM_2,columns_to_drop)
    df_80_O_KM_4_C=df_80_O_KM_4.copy()
    df_80_O_KM_4_WC=drop_columns_from_df(df_80_O_KM_4,columns_to_drop)
    df_80_O_KM_8_C=df_80_O_KM_8.copy()
    df_80_O_KM_8_WC=drop_columns_from_df(df_80_O_KM_8,columns_to_drop)

    df_80_WO_KM_0_C=df_80_WO_KM_0.copy()
    df_80_WO_KM_0_WC=drop_columns_from_df(df_80_WO_KM_0,columns_to_drop)
    df_80_WO_KM_2=df_80_WO_KM_0.copy()
    df_80_WO_KM_2_WC=drop_columns_from_df(df_80_WO_KM_2,columns_to_drop)
    df_80_WO_KM_4_C=df_80_WO_KM_4.copy()
    df_80_WO_KM_4_WC=drop_columns_from_df(df_80_WO_KM_4,columns_to_drop)
    df_80_WO_KM_8_C=df_80_WO_KM_8.copy()
    df_80_WO_KM_8_WC=drop_columns_from_df(df_80_WO_KM_8,columns_to_drop)

    df_75_O_KM_0_C=df_75_O_KM_0.copy()
    df_75_O_KM_0_WC=drop_columns_from_df(df_75_O_KM_0,columns_to_drop)
    df_75_O_KM_2=df_75_O_KM_0.copy()
    df_75_O_KM_2_WC=drop_columns_from_df(df_75_O_KM_2,columns_to_drop)
    df_75_O_KM_4_C=df_75_O_KM_4.copy()
    df_75_O_KM_4_WC=drop_columns_from_df(df_75_O_KM_4,columns_to_drop)
    df_75_O_KM_8_C=df_75_O_KM_8.copy()
    df_75_O_KM_8_WC=drop_columns_from_df(df_75_O_KM_8,columns_to_drop)

    df_75_WO_KM_0_C=df_75_WO_KM_0.copy()
    df_75_WO_KM_0_WC=drop_columns_from_df(df_75_WO_KM_0,columns_to_drop)
    df_75_WO_KM_2=df_75_WO_KM_0.copy()
    df_75_WO_KM_2_WC=drop_columns_from_df(df_75_WO_KM_2,columns_to_drop)
    df_75_WO_KM_4_C=df_75_WO_KM_4.copy()
    df_75_WO_KM_4_WC=drop_columns_from_df(df_75_WO_KM_4,columns_to_drop)
    df_75_WO_KM_8_C=df_75_WO_KM_8.copy()
    df_75_WO_KM_8_WC=drop_columns_from_df(df_75_WO_KM_8,columns_to_drop)

    #ME
    df_80_O_ME_0_C=df_80_O_ME_0.copy()
    df_80_O_ME_0_WC=drop_columns_from_df(df_80_O_ME_0,columns_to_drop)
    df_80_O_ME_2=df_80_O_ME_0.copy()
    df_80_O_ME_2_WC=drop_columns_from_df(df_80_O_ME_2,columns_to_drop)
    df_80_O_ME_4_C=df_80_O_ME_4.copy()
    df_80_O_ME_4_WC=drop_columns_from_df(df_80_O_ME_4,columns_to_drop)
    df_80_O_ME_8_C=df_80_O_ME_8.copy()
    df_80_O_ME_8_WC=drop_columns_from_df(df_80_O_ME_8,columns_to_drop)

    df_80_WO_ME_0_C=df_80_WO_ME_0.copy()
    df_80_WO_ME_0_WC=drop_columns_from_df(df_80_WO_ME_0,columns_to_drop)
    df_80_WO_ME_2=df_80_WO_ME_0.copy()
    df_80_WO_ME_2_WC=drop_columns_from_df(df_80_WO_ME_2,columns_to_drop)
    df_80_WO_ME_4_C=df_80_WO_ME_4.copy()
    df_80_WO_ME_4_WC=drop_columns_from_df(df_80_WO_ME_4,columns_to_drop)
    df_80_WO_ME_8_C=df_80_WO_ME_8.copy()
    df_80_WO_ME_8_WC=drop_columns_from_df(df_80_WO_ME_8,columns_to_drop)

    df_75_O_ME_0_C=df_75_O_ME_0.copy()
    df_75_O_ME_0_WC=drop_columns_from_df(df_75_O_ME_0,columns_to_drop)
    df_75_O_ME_2=df_75_O_ME_0.copy()
    df_75_O_ME_2_WC=drop_columns_from_df(df_75_O_ME_2,columns_to_drop)
    df_75_O_ME_4_C=df_75_O_ME_4.copy()
    df_75_O_ME_4_WC=drop_columns_from_df(df_75_O_ME_4,columns_to_drop)
    df_75_O_ME_8_C=df_75_O_ME_8.copy()
    df_75_O_ME_8_WC=drop_columns_from_df(df_75_O_ME_8,columns_to_drop)

    df_75_WO_ME_0_C=df_75_WO_ME_0.copy()
    df_75_WO_ME_0_WC=drop_columns_from_df(df_75_WO_ME_0,columns_to_drop)
    df_75_WO_ME_2=df_75_WO_ME_0.copy()
    df_75_WO_ME_2_WC=drop_columns_from_df(df_75_WO_ME_2,columns_to_drop)
    df_75_WO_ME_4_C=df_75_WO_ME_4.copy()
    df_75_WO_ME_4_WC=drop_columns_from_df(df_75_WO_ME_4,columns_to_drop)
    df_75_WO_ME_8_C=df_75_WO_ME_8.copy()
    df_75_WO_ME_8_WC=drop_columns_from_df(df_75_WO_ME_8,columns_to_drop)

    #NO
    df_80_O_NO_0_C=df_80_O_NO_0.copy()
    df_80_O_NO_0_WC=drop_columns_from_df(df_80_O_NO_0,columns_to_drop)
    df_80_O_NO_2=df_80_O_NO_0.copy()
    df_80_O_NO_2_WC=drop_columns_from_df(df_80_O_NO_2,columns_to_drop)
    df_80_O_NO_4_C=df_80_O_NO_4.copy()
    df_80_O_NO_4_WC=drop_columns_from_df(df_80_O_NO_4,columns_to_drop)
    df_80_O_NO_8_C=df_80_O_NO_8.copy()
    df_80_O_NO_8_WC=drop_columns_from_df(df_80_O_NO_8,columns_to_drop)

    df_80_WO_NO_0_C=df_80_WO_NO_0.copy()
    df_80_WO_NO_0_WC=drop_columns_from_df(df_80_WO_NO_0,columns_to_drop)
    df_80_WO_NO_2=df_80_WO_NO_0.copy()
    df_80_WO_NO_2_WC=drop_columns_from_df(df_80_WO_NO_2,columns_to_drop)
    df_80_WO_NO_4_C=df_80_WO_NO_4.copy()
    df_80_WO_NO_4_WC=drop_columns_from_df(df_80_WO_NO_4,columns_to_drop)
    df_80_WO_NO_8_C=df_80_WO_NO_8.copy()
    df_80_WO_NO_8_WC=drop_columns_from_df(df_80_WO_NO_8,columns_to_drop)

    df_75_O_NO_0_C=df_75_O_NO_0.copy()
    df_75_O_NO_0_WC=drop_columns_from_df(df_75_O_NO_0,columns_to_drop)
    df_75_O_NO_2=df_75_O_NO_0.copy()
    df_75_O_NO_2_WC=drop_columns_from_df(df_75_O_NO_2,columns_to_drop)
    df_75_O_NO_4_C=df_75_O_NO_4.copy()
    df_75_O_NO_4_WC=drop_columns_from_df(df_75_O_NO_4,columns_to_drop)
    df_75_O_NO_8_C=df_75_O_NO_8.copy()
    df_75_O_NO_8_WC=drop_columns_from_df(df_75_O_NO_8,columns_to_drop)

    df_75_WO_NO_0_C=df_75_WO_NO_0.copy()
    df_75_WO_NO_0_WC=drop_columns_from_df(df_75_WO_NO_0,columns_to_drop)
    df_75_WO_NO_2=df_75_WO_NO_0.copy()
    df_75_WO_NO_2_WC=drop_columns_from_df(df_75_WO_NO_2,columns_to_drop)
    df_75_WO_NO_4_C=df_75_WO_NO_4.copy()
    df_75_WO_NO_4_WC=drop_columns_from_df(df_75_WO_NO_4,columns_to_drop)
    df_75_WO_NO_8_C=df_75_WO_NO_8.copy()
    df_75_WO_NO_8_WC=drop_columns_from_df(df_75_WO_NO_8,columns_to_drop)

    
