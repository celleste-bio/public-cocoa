import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir
from utils import read_yaml

sys.path.append("/home/public-cocoa/src/prep")
from clean_data import clean_test_data
from Kmeans_missing_value import clustering_and_replace_missing_values
from Median_and_Frequent_Function import fill_missing_values_median_and_mode
from Normal_and_Frequent_Function import Normal_and_Frequent_Function
from outliers_by_IQR import IQR_outliers
from hirarcial_tree_column import hirrarcial_tree
from dummies_order import dummies_order

sys.path.append("/home/public-cocoa/src/voting")
from main_voting import main_voting

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

def drop_columns_from_df(df, columns_to_drop):
    df.drop(columns=columns_to_drop, axis=1, inplace=True)
    return df

def preperation_for_model(df, method,apply_outliers ,Hierarchical_tree_clusters,C_OR_WC):
    # script_path = "/home/public-cocoa/src/the_prediction/main.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_prediction.yaml")
    config = read_yaml(config_path)
    df = clean_test_data(df)
    if apply_outliers :
        df = IQR_outliers(df)

    if method == "KM":
        df = clustering_and_replace_missing_values(df, 3)
    elif method == "ME":
        df = fill_missing_values_median_and_mode(df)
    elif method == "NO":
        df = Normal_and_Frequent_Function(df)
    df_dummies = dummies_order(df)
    df_dummies.columns = df_dummies.columns.str.replace('<=', 'lte', regex=False)
    df_dummies.columns = df_dummies.columns.str.replace('>', 'gt', regex=False)
    df_dummies = hirrarcial_tree(df_dummies, Hierarchical_tree_clusters)

    
    if C_OR_WC == "WC":
        df_dummies = drop_columns_from_df(df_dummies, config["columns_cotyledon"])




    return df_dummies

def main():
    # script_path = "/home/public-cocoa/src/the_prediction/main.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_prediction.yaml")
    config = read_yaml(config_path)

    #'80_WO_KM_8_WC'
    # Reading the csv file
    data80 = read_single_csv(config['train_data_80'])
    data20 = read_single_csv(config['test_data_20'])
    df = data20.copy()
    method = 'KM'
    apply_outliers  = True
    Hierarchical_tree_clusters = 8
    C_OR_WC = 'WC'
    df = preperation_for_model(df, method,apply_outliers ,Hierarchical_tree_clusters,C_OR_WC )



    output_dir = config["output_directory"]
    os.makedirs(output_dir, exist_ok=True)
    # Construct the output filename
    output_filename = f"20_{C_OR_WC}_{method}_{Hierarchical_tree_clusters}.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    for column in data80.columns:
        if column not in df.columns:
            df[column] = 0
            print(column)

    for column in df.columns:
        if column not in data80.columns:
            print("- ", column)

    y = df[config['target']]  # Target variable
    X = df.drop(config['target'], axis=1)  # Features

    # Convert to NumPy arrays
    X = X.values
    y = y.values 

    model = joblib.load("/home/public-cocoa/data/joblib/voting_regressor_model.pkl")



    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)


    # Print cross-validation scores
    print(f'Cross-validation scores (MAE): {mae}')

    print(f'Cross-validation scores (MSE): {mse}')

    print(f'Cross-validation scores (MAPE): {mape}')

    # Save cross-validation scores to CSV
    scores_dict = {
        'MAE': [mae],
        'MSE': [mse],
        'MAPE': [mape]
    }


    scores_df = pd.DataFrame(scores_dict)
    csv_output_path = os.path.join(config["output_directory"], "cross_validation_scores.csv")
    scores_df.to_csv(csv_output_path, index=False)

if __name__ == "__main__":
    main()