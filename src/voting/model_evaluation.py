from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import joblib
import os
import sys

sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir
from utils import read_yaml

sys.path.append("/home/public-cocoa/src/prep")
from main_new import prep_data
from dummies_order import dummies_order
from fill_missing_values import fill_missing_values
from hirarcial_tree_column import hirrarcial_tree
from Kmeans_missing_value import clustering_and_replace_missing_values

def read_single_csv(directory_path):
    file_path = directory_path
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while reading the file '{file_path}': {e}")


def main_model_evalution():
    # script_path = "/home/public-cocoa/src/voting/model_evaluation.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_main_voting.yaml")
    config = read_yaml(config_path)

    # Reading the csv file
    data20 = read_single_csv(config['test20_directory'])
    df = data20.copy()


    model = joblib.load(config["load_voting_regressor"])
    scalar = joblib.load(config["load_scalar"])

  
    method="KM"
    outliers= True
    Hierarchical_tree_clusters = 8
    C_OR_WC = "wc"

    cleaned_data=prep_data(df,method,outliers,Hierarchical_tree_clusters,C_OR_WC)
    #the data is clean without duplication and without null in target column
    df_dummies = hirrarcial_tree(cleaned_data, Hierarchical_tree_clusters)
    df_dummies.info()
    y = df_dummies[config['target']]
    X = df_dummies.drop(columns=[config['target']], axis=1)
 
    # Convert to NumPy arrays
    cleaned_data = cleaned_data.values
    y = y.values  
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)


    # Print cross-validation scores
    print(f'Cross-validation scores (MAE): {mae}')

    print(f'Cross-validation scores (MSE): {mse}')

    print(f'Cross-validation scores (MAPE): {mape}')



if __name__ == "__main__":
    main_model_evalution()