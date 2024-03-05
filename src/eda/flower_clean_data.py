
"""
clean flower data

"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt


def go_back_dir(path, number):
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def create_path(script_path):
    # context
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    return(db_path)
 
def cleaning(connection):

    query = """
    SELECT *
    FROM flower
    """

    result = pd.read_sql_query(query, connection)
    result.info()
    result.head()

    numeric_columns=result.select_dtypes(include=['number']).columns
    # List of all columns
    all_columns = result.columns
    # Identify non-numeric columns
    categorical_columns = [col for col in all_columns if col not in numeric_columns]
    # Replace "-" with NaN in both numerical and categorical columns
    result[numeric_columns] = result[numeric_columns].replace('-', np.nan)
    result[categorical_columns] = result[categorical_columns].replace('-', np.nan)
    # Convert numerical columns to float
    result[numeric_columns] = result[numeric_columns].astype(float)
    #-----------------------------------------------------
    missing_percentage = result.isnull().mean() * 100
    # Create a DataFrame to display the results
    missing_info = pd.DataFrame({
        'Column': missing_percentage.index,
        'Missing Percentage': missing_percentage.values })
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(missing_info['Column'], missing_info['Missing Percentage'], color='skyblue')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage')
    plt.title('flower')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Show the plot
    plt.show()

    missing_percent = (result.isnull().mean() * 100).round(2)

    # Filter out columns with more than 80% missing values
    columns_to_drop = missing_percent[missing_percent > 80].index

    # Drop the columns with more than 80% missing values
    result = result.drop(columns=columns_to_drop)

    # Print the remaining columns
    print("Remaining columns after dropping columns with more than 80% missing values:")
    print(result.columns)

def left_join(connection):
    # Define your SQL query with LEFT JOIN
    query_join_tables = """
    SELECT *
    FROM butterfat
    LEFT JOIN bean ON butterfat.[Clone name] = bean.[Clone name] AND butterfat.[Refcode] = bean.[Refcode]
    WHERE butterfat.fat IS NOT NULL
    """
    # Use read_sql_query with the connection 
    result = pd.read_sql_query(query_join_tables, connection)
    result.info()
    result.head()

    


def main():
    script_path="/home/public-cocoa/src/eda/flower_clean_data.py"
    #create database
    #script_path = os.path.realpath(__file__)
    db_path=create_path(script_path)
    #Create a SQLite database connection
    connection = sqlite.connect(db_path)

    connection.close()

if __name__ == "__main__":
    main()