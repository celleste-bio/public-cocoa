import pandas as pd
import sys
import yaml
import os

sys.path.append("/home/public-cocoa/src/")
from clean_data import clean_data
sys.path.append("/home/public-cocoa/src/prep/")
from path_utils import go_back_dir

def get_configs(script_path):
    script_dir = go_back_dir(script_path, 0)
    config_path = os.path.join(script_dir, "config_dummies.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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

    print("Converted columns to float:", numeric_cols)

    # Check the data types of the DataFrame after conversion
    print(df.dtypes)
    return df

def dummies_order_func():
    # script_path= "/home/public-cocoa/src/prep/dummies_order.py"
    script_path = os.path.realpath(__file__)
    config = get_configs(script_path)
    id_columns = config['id_columns']
    columns_to_drop = [config['columns_to_drop']]
    df = clean_data()
    # Replace values in columns with meaningful order
    for column, mapping in config['column_mappings'].items():
        df[column] = df[column].replace(mapping)
    df = drop_columns_from_df(df, columns_to_drop)
    df = drop_columns_from_df(df, id_columns)
    df.info()
    df = convert_to_float(df)
    dummies = df.select_dtypes(include=object).columns
    df = pd.get_dummies(df, columns=dummies)
    df[df.select_dtypes(include=bool).columns] = df[df.select_dtypes(include=bool).columns].astype(int)
    df.info()

    return df


if __name__ == "__main__":
    dummies_order_func()