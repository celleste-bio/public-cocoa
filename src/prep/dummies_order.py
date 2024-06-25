import pandas as pd
import sys
import yaml
import os

sys.path.append("/home/public-cocoa/src/")
from path_utils import go_back_dir
from utils import read_yaml

# Function to drop columns from DataFrame
def drop_columns_from_df(df, columns_to_drop):
    df = df.drop(columns=columns_to_drop, errors='ignore')
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
    return df

def dummies_order(data):
    script_path = os.path.realpath(__file__)
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_dummies.yaml")
    config = read_yaml(config_path)
    
    # Replace values in columns with meaningful order
    for column, mapping in config['column_mappings'].items():
        data[column] = data[column].replace(mapping)
    data = convert_to_float(data)
    dummies = data.select_dtypes(include=object).columns
    data = pd.get_dummies(data, columns=dummies)
    data[data.select_dtypes(include=bool).columns] = data[data.select_dtypes(include=bool).columns].astype(int)
    return data

if __name__ == "__main__":
    dummies_order()