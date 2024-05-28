"""
clean ICGD
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from scipy.stats import norm,f_oneway
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats

sys.path.append("/home/public-cocoa/src/")
from path_utils import go_back_dir


def connect_the_container(script_path):
    
    project_path = go_back_dir(script_path, 2)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    return connection

def replace_dash_with_nan(df):
    # Replace "-" with NaN in both numerical and categorical columns
    df = df.replace('-', np.nan)
    return df


def calculate_prc_missing_values(df):
    missing_percent = (df.isnull().mean() * 100).round(2)
    # Create a DataFrame to display the results
    missing_info = pd.DataFrame({
      'Column': missing_percent.index,
      'Missing Percentage': missing_percent.values })
    
    # Filter out columns with more than 70% missing values
    columns_to_drop = missing_percent[missing_percent > 70].index
    # Drop the columns with more than 70% missing values
    df = df.drop(columns=columns_to_drop)
    df.dropna(subset=['Bean_Total wet weight'], inplace=True)
    
    return df

def create_data_frame(conn):
    # Fetch data from each table
    table_flower = pd.read_sql_query("SELECT * FROM flower", conn)
    table_fruit = pd.read_sql_query("SELECT * FROM fruit", conn)
    table_bean = pd.read_sql_query("SELECT * FROM bean", conn)

    # Function to add table name prefix to columns, except for "Refcode" and "Clone name"
    def add_table_prefix(df, table_name):
        prefixed_columns = {col: f"{table_name}_{col}" if col not in ["Refcode", "Clone name"] else col for col in df.columns}
        return df.rename(columns=prefixed_columns)

    # Rename columns with table names
    table_flower = add_table_prefix(table_flower, 'Flower')
    table_fruit = add_table_prefix(table_fruit, 'Fruit')
    table_bean = add_table_prefix(table_bean, 'Bean')

    # Merge dataframes
    # Inner join all tables on 'Clone name' and 'Refcode'
    merged_df = pd.merge(table_flower, table_fruit, 
                     left_on=['Clone name', 'Refcode'], 
                     right_on=['Clone name', 'Refcode'], 
                     how='inner')
    merged_df = pd.merge(merged_df, table_bean, 
                     left_on=['Clone name', 'Refcode'], 
                     right_on=['Clone name', 'Refcode'], 
                     how='inner')

    # Remove rows with duplicate ["Refcode", "Clone name"]
    merged_df = merged_df.drop_duplicates(subset=["Refcode", "Clone name"])

    return merged_df



def main():

    # script_path="/home/public-cocoa/src/eda/clean_data.py"
    script_path = os.path.realpath(__file__)
    connection = connect_the_container(script_path)

    #create dataFrame
    #and drop every duplicated clone name + refcode
    dataFrame = create_data_frame(connection)
    dataFrame.info()
    #replace dash with nan
    dataFrame = replace_dash_with_nan(dataFrame)
    dataFrame.info()
    #calculate missing value and drop every coulmn with more than 70% missing value
    #and drop every row that has nan in Bean_Total wet weight  
    dataFrame = calculate_prc_missing_values(dataFrame)
    dataFrame.info()


    connection.close()

if __name__ == "__main__":
    main()