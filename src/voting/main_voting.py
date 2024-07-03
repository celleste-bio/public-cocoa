import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import os
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb  # Using cb alias for CatBoost
import joblib  # Import joblib here


sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir
from utils import read_yaml

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

# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_test, y_pred):
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

def train(X_train, y_train, X_test, y_test, config):
    
    # Extract the necessary parameters from the configuration
    data_directory = config['data_directory']
    target = config['target']
    rf_params = config['RandomForestRegressor']
    huber_params = config['HuberRegressor']
    mlp_params = config['MLPRegressor']
    dt_params = config['DecisionTreeRegressor']
    gbr_params = config['GradientBoostingRegressor']
    xgb_params = config['XGBRegressor']
    lgb_params = config['LGBMRegressor']
    cat_params = config['CatBoostRegressor']
    svr_params = config['SVR']
    voting_params = config['VotingRegressor']

    # Create the VotingRegressor
    # Initialize models with the extracted hyperparameters
    reg1 = RandomForestRegressor(**rf_params)
    reg2 = HuberRegressor(**huber_params)
    reg3 = MLPRegressor(**mlp_params)
    reg4 = DecisionTreeRegressor(**dt_params)
    reg5 = GradientBoostingRegressor(**gbr_params)
    reg6 = xgb.XGBRegressor(**xgb_params)
    reg7 = lgb.LGBMRegressor(**lgb_params)
    reg8 = cb.CatBoostRegressor(**cat_params)
    reg9 = SVR(**svr_params)

    # Initialize VotingRegressor with the actual estimator instances and weights from the config
    estimators = [
        ('rf', reg1), ('huber', reg2), ('mlp', reg3), ('dt', reg4),
        ('gbr', reg5), ('xgb', reg6), ('lgb', reg7), ('cat', reg8), ('svr', reg9)
    ]
    weights = config['VotingRegressor']['weights']
    voting_reg = VotingRegressor(estimators=estimators, weights=weights)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # Initialize lists to store scores
    mae_scores = []
    mse_scores = []
    mape_scores = []

        # Perform cross-validation manually to compute multiple metrics
    for train_index, test_index in cv.split(X_train):
        X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        scaler = MinMaxScaler()
        X_train_cv = scaler.fit_transform(X_train_cv)
        X_test_cv = scaler.transform(X_test_cv)

        voting_reg.fit(X_train_cv, y_train_cv)
        y_pred = voting_reg.predict(X_test_cv)

        mae = mean_absolute_error(y_test_cv, y_pred)
        mse = mean_squared_error(y_test_cv, y_pred)
        mape = mean_absolute_percentage_error(y_test_cv, y_pred)

        mae_scores.append(mae)
        mse_scores.append(mse)
        mape_scores.append(mape)

    # Print cross-validation scores
    print(f'Cross-validation scores (MAE): {mae_scores}')
    print(f'Mean cross-validation score (MAE): {np.mean(mae_scores)}')

    print(f'Cross-validation scores (MSE): {mse_scores}')
    print(f'Mean cross-validation score (MSE): {np.mean(mse_scores)}')

    print(f'Cross-validation scores (MAPE): {mape_scores}')
    print(f'Mean cross-validation score (MAPE): {np.mean(mape_scores)}')

    # Fit the voting regressor on the entire training dataset
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)

    # Create output directory if it does not exist
    output_dir = config["save_praped_data"]
    os.makedirs(output_dir, exist_ok=True)

    # Save the min-max scaler to a file using joblib
    min_max_scaler_filename = os.path.join(output_dir, 'min_max_scaler.pkl')
    joblib.dump(scaler, min_max_scaler_filename)

    # Save the trained model to a file using joblib
    model_filename = os.path.join(output_dir, 'voting_regressor_model.pkl')
    joblib.dump(voting_reg, model_filename)

    print(f'Trained model saved as {model_filename}')


def main_voting():
    # script_path = "/home/public-cocoa/src/voting/main.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_main_voting.yaml")
    config = read_yaml(config_path)

    # Reading the csv file
    data = read_single_csv(config['data_directory'])
    df = data.copy()
    y = df[config['target']]
    X = df.drop(columns=[config['target']], axis=1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model and evaluate performance
    train(X_train, y_train, X_test, y_test, config)


   

    

if __name__ == "__main__":
    main_voting()