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
    # Initialize the models using the loaded configuration
    rf_model = RandomForestRegressor(**config["RandomForestRegressor"])
    knn_model = KNeighborsRegressor(**config["KNeighborsRegressor"])
    huber_model = HuberRegressor(**config["HuberRegressor"])
    nn_model = MLPRegressor(**config["MLPRegressor"])

    # Create the VotingRegressor
    voting_regressor = VotingRegressor(
        estimators=[
            ('rf', rf_model),
            ('knn', knn_model),
            ('huber', huber_model),
            ('nn', nn_model),
        ]
    )

    # Fit the ensemble model on the training data
    voting_regressor.fit(X_train, y_train)

    # Predict the test set
    y_pred = voting_regressor.predict(X_test)

    # Evaluate the performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
  
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}')

    return voting_regressor

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
    model = train(X_train, y_train, X_test, y_test, config)

    return model

    

if __name__ == "__main__":
    main_voting()