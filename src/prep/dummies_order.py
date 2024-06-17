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
    df = df.drop_duplicates(subset=id_columns)
    df = drop_columns_from_df(df, columns_to_drop)
    df = drop_columns_from_df(df, id_columns)
    df.info()
    

    df = pd.get_dummies(df, columns=dummies)


    






dummies = ['bean_shape', 'flower_ligule_colour', 'fruit_shape', 'fruit_colour', 'fruit_basal_constriction', 'fruit_apex_form', 'fruit_furrow_desc']



bool_columns = df.select_dtypes(include=bool).columns
object_columns = df.select_dtypes(include = object).columns
df[bool_columns] = df[bool_columns].astype(int)
df[object_columns] = df[object_columns].astype(float)


# Assuming 'df' is your DataFrame and 'target_column' is defined
target_column = "Total wet weight"

# Selecting object columns excluding the target column
object_columns = df.select_dtypes(include='object').columns
object_columns = object_columns[object_columns != target_column]

# Now 'object_columns' contains all object-type columns except the target column
print(object_columns)


# Convert all object columns to float
for col in df.select_dtypes(include=['object']).columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except ValueError:
        print(f"Could not convert column {col} to float.")
df_no_duplicates = df.drop_duplicates(subset=['clone_name', 'refcode'])

# Print the updated DataFrame
print(df_no_duplicates)
df_no_duplicates.info()
print(f"Original DataFrame shape: {df.shape}")
print(f"DataFrame shape after dropping duplicates: {df_no_duplicates.shape}")