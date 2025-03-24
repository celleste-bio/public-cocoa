import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir
from utils import read_yaml

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_best_model(model, hyperparams, X, y, cv=5):
    mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    scoring = {
    'mse': make_scorer(mean_squared_error, greater_is_better=False),
    'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    'mape': mape_scorer,
    'r2': make_scorer(r2_score)
    }

    grid_search = GridSearchCV(estimator=model, param_grid=hyperparams, cv=cv, scoring=scoring,refit='mape')
    grid_search.fit(X, y)
    best_index = grid_search.best_index_
    cv_results = grid_search.cv_results_
    # Define the metrics and their corresponding keys in cv_results
    metrics = {
    'mae': 'mean_test_mae',
    'mse': 'mean_test_mse',
    'mape': 'mean_test_mape',
    'r2': 'mean_test_r2'
    }

    # Dictionary to store results
    results_dict = {}
    # Iterate over metrics and extract values at best_index
    for metric_name, cv_key in metrics.items():
        results_dict[metric_name] = cv_results[cv_key][best_index]
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print('Best Parameters:',best_model)
    print("Best Parameters:", best_params)
    print("Best mape_scorer:", results_dict)
    
    return best_model, best_params,results_dict

def total_average_score(scoring_list, metric):
    number_metrics_scoring = len(scoring_list)
    sum_scoring = sum(evaluation[metric] for evaluation in scoring_list)
    average = sum_scoring / number_metrics_scoring
    return average * -1


def read_csv_files_into_dict(directory_path):
    all_files = os.listdir(directory_path)
    csv_files = [f for f in all_files if f.endswith('.csv')]
    df_dict = {}
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        try:
            df = pd.read_csv(file_path)
            # Use the file name without the '.csv' extension as the dictionary key
            df_dict[os.path.splitext(file)[0]] = df
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print(f"An error occurred while reading the file '{file_path}': {e}")
    return df_dict
    
def main():

    # script_path = "/home/public-cocoa/src/grid_reseach/main_new.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_grid_reseach.yaml")
    config = read_yaml(config_path)
    
    # Read all CSV files from the specified directory into a dictionary
    df_dict = read_csv_files_into_dict(config['data_directory'] )

    df_columns = df_dict['80_O_KM_4_WC']
    print("Columns of DataFrame:", df_columns)

    df_columns.info()

    # Assuming df_dict and config are already defined
    eval_results = {}

    for key, df in df_dict.items():
        if key not in ['25_test.csv', '20_test.csv']:
            y = df[config['target']]
            X = df.drop(columns=[config['target']], axis=1)

            # Min-Max Normalization
            scaler = MinMaxScaler()
            df_min_max = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Model training and evaluation
            models = config['param_grids']
            model_results = []  # Initialize a list to store results for each model

            for model_config in models:
                model_name = model_config['name']
                model_class = globals()[model_name]
                hyperparams = model_config['params']

                # Assuming get_best_model is a defined function that returns the best model along with its parameters and scores
                best_model, best_params, best_score = get_best_model(model_class(), hyperparams, df_min_max, y)

                model_results.append({
                    'Model': model_name,
                    'best_params': best_params,
                    'MAE': best_score['mae'],
                    'MSE': best_score['mse'],
                    'MAPE': best_score['mape'],
                    'R2': best_score['r2']
                })

            # Store the model results corresponding to the current key (CSV file identifier)
            eval_results[key] = model_results

    # Create the output directory if it doesn't exist
    output_dir = config["save_praped_data"]
    os.makedirs(output_dir, exist_ok=True)

    # Write each DataFrame to a separate CSV file
    # Write the DataFrame to a single CSV file
    csv_path = os.path.join(output_dir, "all_results.csv")
    eval_results.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()








   