import os
import sys
import pandas as pd

sys.path.append("/home/public-cocoa/src")
from path_utils import go_back_dir
from utils import read_yaml

sys.path.append("/home/public-cocoa/src/prep")
from main_new import prep_data

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



def main():
    # script_path = "/home/public-cocoa/src/the_prediction/main.py"
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_prediction.yaml")
    config = read_yaml(config_path)

    #'80_WO_KM_8_WC'
    # Reading the csv file
    data = read_single_csv(config['test_data_20'])
    df = data.copy()
    method = 'KM'
    outliers = True
    Hierarchical_tree_clusters = 8
    C_OR_WC = 'WC'
    df = prep_data(df, method,outliers,Hierarchical_tree_clusters,C_OR_WC )
    y = df[config['target']]
    X = df.drop(columns=[config['target']], axis=1)

    # Reading the new data for prediction
    new_data = read_single_csv(config['new_data_directory'])

    model = main_voting()
    # Make predictions on the new data
    predictions = model.predict(new_data)

    # Add the predictions to the new data DataFrame
    new_data[config['target']] = predictions

    # Save the new data with predictions to a CSV file
    output_path = config['output_directory']
    new_data.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()