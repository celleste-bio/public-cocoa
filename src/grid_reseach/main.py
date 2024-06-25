from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import make_scorer
from sklearn.linear_model import HuberRegressor
import joblib 
import tensorflow as tf
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import sys

sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir
from utils import read_yaml

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def read_file(path):
    try:
        # Read the CSV file
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

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

def Total_average_score(scoring_list,metric):
    number_Metrics_Scoring=len(scoring_list)
    sum_Scoring=0
    for evaluation in scoring_list:
        sum_Scoring+=evaluation[metric]
    average=sum_Scoring/number_Metrics_Scoring
    return average*-1 

def write_to_excel(df,Subject):
        # Write the evaluation results to a CSV file
        df.to_csv(f'{Subject}.csv', index=False)
        print("Evaluation results saved to 'evaluation_results.csv'")

def main():

    # script_path="/home/public-cocoa/src/grid_reseach/main.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_grid_reseach.yaml")
    config = read_yaml(config_path)


    df = read_file(script_path)
    df.info()

    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Total wet weight bean','Clone name + Refcode'])  # Features
    y = df['Total wet weight bean'] 

    ## Min-Max Normalization
    min_max_scaler = MinMaxScaler()
    df_min_max = X.copy()

    #this action below is to save the data frame
    df_min_max[X.columns] = min_max_scaler.fit_transform(X)

    # Model training
    knn = KNeighborsRegressor()
    best_knn_model, best_knn_params,best_knn_score = get_best_model(knn, param_grid_knn,df_min_max, y, cv=5)
    rf = RandomForestRegressor()
    best_rf_model, best_rf_params, best_rf_score = get_best_model(rf, param_grid_rf, df_min_max, y, cv=5)
    huber = HuberRegressor()
    best_huber_model, best_huber_params, best_huber_score = get_best_model(huber, param_grid_huber, df_min_max, y, cv=5)
    mlp = MLPRegressor()
    best_mlp_model, best_mlp_params, best_mlp_score = get_best_model(mlp, param_grid_mlp, df_min_max, y, cv=5)
    dt = DecisionTreeRegressor()
    best_dt_model, best_dt_params, best_dt_score = get_best_model(dt, param_grid_dt, df_min_max, y, cv=5)
    gb = GradientBoostingRegressor()
    best_gb_model, best_gb_params, best_gb_score = get_best_model(gb, param_grid_gb, df_min_max, y, cv=5)
    xgb = XGBRegressor()
    best_xgb_model, best_xgb_params, best_xgb_score = get_best_model(xgb, param_grid_xgb, df_min_max, y, cv=5)
    lgbm = LGBMRegressor()
    best_lgbm_model, best_lgbm_params, best_lgbm_score = get_best_model(lgbm, param_grid_lgbm, df_min_max, y, cv=5)
    cat = CatBoostRegressor()
    best_cat_model, best_cat_params, best_cat_score = get_best_model(cat, param_grid_cat, df_min_max, y, cv=5)
    svr = SVR()
    best_svr_model, best_svr_params, best_svr_score = get_best_model(svr, param_grid_svr, df_min_max, y, cv=5)
        

    eval_results = pd.DataFrame({
        'Model': ['knn', 'Random Forest', 'Huber Regression'],
        'param_grid': [param_grid_knn, param_grid_rf, param_grid_huber],
        'best_params': [best_knn_params, best_rf_params, best_huber_params],
        'MAE': [best_knn_score['mae'], best_rf_score['mae'], best_huber_score['mae']],
        'MSE':[best_knn_score['mse'], best_rf_score['mse'], best_huber_score['mse']],
        'MAPE':[best_knn_score['mape'], best_rf_score['mape'], best_huber_score['mape']],
        'R2':[best_knn_score['r2'], best_rf_score['r2'], best_huber_score['r2']],
        'File_name':[file_name,file_name,file_name]})
        
    eval_results.loc[len(eval_results)] = ['MLP Regressor', param_grid_mlp, best_mlp_params, best_mlp_score['mae'],best_mlp_score['mse'],best_mlp_score['mape'],best_mlp_score['r2'],file_name]
    eval_results.loc[len(eval_results)] = ['Decision Tree', param_grid_dt, best_dt_params, best_dt_score['mae'],best_dt_score['mse'],best_dt_score['mape'],best_dt_score['r2'],file_name]
    eval_results.loc[len(eval_results)] = ['Gradient Boosting', param_grid_gb, best_gb_params,best_gb_score['mae'],best_gb_score['mse'],best_gb_score['mape'],best_gb_score['r2'],file_name]
    eval_results.loc[len(eval_results)] = ['XGBoost', param_grid_xgb, best_xgb_params, best_xgb_score['mae'],best_xgb_score['mse'],best_xgb_score['mape'],best_xgb_score['r2'],file_name]
    eval_results.loc[len(eval_results)] = ['LightGBM', param_grid_lgbm, best_lgbm_params, best_lgbm_score['mae'],best_lgbm_score['mse'],best_lgbm_score['mape'],best_lgbm_score['r2'],file_name]
    eval_results.loc[len(eval_results)] = ['CatBoost', param_grid_cat, best_cat_params, best_cat_score['mae'], best_cat_score['mse'], best_cat_score['mape'],best_cat_score['r2'],file_name]
    eval_results.loc[len(eval_results)] = ['SVR', param_grid_svr, best_svr_params, best_svr_score['mae'],best_svr_score['mse'],best_svr_score['mape'],best_svr_score['r2'],file_name]
    scoring_list=[best_knn_score,best_rf_score,best_huber_score,best_mlp_score,best_dt_score,best_gb_score,best_xgb_score,best_lgbm_score,best_cat_score,best_svr_score]
    df_average_all_model[len(df_average_all_model)]=[file_name,Total_average_score(scoring_list,'mae'),Total_average_score(scoring_list,'mse'),Total_average_score(scoring_list,'mape'),Total_average_score(scoring_list,'r2')]
    Total_evaluation_Models=pd.concat([Total_evaluation_Models,eval_results], ignore_index=True)

    write_to_excel(Total_evaluation_Models,'Parameter_evaluation_results')
    write_to_excel(df_average_all_model,'average_Metric_per_file')
    

if __name__ == "__main__":
    main()