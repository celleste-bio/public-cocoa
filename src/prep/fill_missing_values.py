import os
import yaml
import sys
sys.path.append("/home/public-cocoa/src/prep")
from Kmeans_missing_value import clustering_and_replace_missing_values
from Median_and_Frequent_Function import fill_missing_values_median_and_mode
from Normal_and_Frequent_Function import Normal_and_Frequent_Function
from outliers_by_IQR import IQR_outliers
from hirarcial_tree_column import hirrarcial_tree
from dummies_order import dummies_order

sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir

def get_configs(script_path, config_file_name):
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, config_file_name)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def drop_columns_from_df(df, columns_to_drop):
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df

def fill_missing_values(df):
    # script_path="/home/public-cocoa/src/prep/fill_missing_values.py"
    script_path = os.path.realpath(__file__)
    config_file_name = "config_missing_values.yaml"
    config_missing_value = get_configs(script_path,config_file_name)

    # Splitting the DataFrame into train and test sets
    train_df_75 = df.sample(frac=0.75, random_state=42)  # 75% for training
    test_df_25 = df.drop(train_df_75.index)

    # Splitting the DataFrame into train and test sets
    train_df_80 = df.sample(frac=0.80, random_state=42)  # 80% for training
    test_df_20 = df.drop(train_df_80.index)
    train_df_80.info()

    # Initialize dictionary to store all DataFrames
    all_dfs = {}
    df_80_O = train_df_80.copy()
    df_80_WO = IQR_outliers(train_df_80)
    df_75_O = train_df_75.copy()
    df_75_WO = IQR_outliers(train_df_75)

  
    # Filling missing values
    all_dfs = {}
    dataframes = {'80_O': df_80_O, '80_WO': df_80_WO, '75_O': df_75_O, '75_WO': df_75_WO}

    for method in config_missing_value['methods']:
        for prefix, df in dataframes.items():
            if method == "KM":
                filled_df = clustering_and_replace_missing_values(df, 3)
            elif method == "ME":
                filled_df = fill_missing_values_median_and_mode(df)
            elif method == "NO":
                filled_df = Normal_and_Frequent_Function(df)
            all_dfs[f"{prefix}_{method}"] = filled_df
    
    for name, df in all_dfs.items():
        df_dummies = dummies_order(df)
        df_dummies.columns = df_dummies.columns.str.replace('<=', 'lte', regex=False)
        df_dummies.columns = df_dummies.columns.str.replace('>', 'gt', regex=False)
        all_dfs[name] = df_dummies

    dummies_20 = dummies_order(test_df_20)
    dummies_20.columns = dummies_20.columns.str.replace('<=', 'lte', regex=False)
    dummies_20.columns = dummies_20.columns.str.replace('>', 'gt', regex=False)
    dummies_25 = dummies_order(test_df_25)
    dummies_25.columns = dummies_25.columns.str.replace('<=', 'lte', regex=False)
    dummies_25.columns = dummies_25.columns.str.replace('>', 'gt', regex=False)

    for name, df in all_dfs.items():
        if name in config_missing_value["file_names_80"]:
            for column in dummies_20.columns:
                if column not in df.columns:
                    df[column] = 0
                    print(column)
        if name in config_missing_value["file_names_75"]:
            for column in dummies_25.columns:
                if column not in df.columns:
                    df[column] = 0
                    print(column)

    for key in list(all_dfs.keys()):
        if key in config_missing_value["file_names_80"]:
            for column in all_dfs[key].columns:
                if column not in dummies_20.columns:
                    test_df_20[column] = 0
                    print(column)
        if key in config_missing_value["file_names_75"]:
            for column in all_dfs[key].columns:
                if column not in dummies_25.columns:
                    test_df_25[column] = 0
                    print(column)


    all_dfs_new={}
    # Hierarchical Clustering
    for key in all_dfs:
        for clusters in config_missing_value["Hierarchical_tree_clusters"]:
            all_dfs_new[f"{key}_{clusters}"] = hirrarcial_tree(all_dfs[key], clusters)

        
    all_dfs_new_final={}

    # Copy DataFrames for 'C' and 'WC' versions
    for key in list(all_dfs_new.keys()):
        all_dfs_new_final[f"{key}_C"] = all_dfs_new[key].copy()
        all_dfs_new_final[f"{key}_WC"] = drop_columns_from_df(all_dfs_new[key], config_missing_value["columns_cotyledon"])
   
    all_dfs_new_final[f"{25}_{"test"}"] = test_df_25
    all_dfs_new_final[f"{20}_{"test"}"] = test_df_20
    return all_dfs_new_final