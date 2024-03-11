
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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

    #dealing with missing value


    # Load your dataset
    result.info()


    # Separate data into two parts: one with missing values and one without
    # Identify features with missing values
    features_with_missing = [ 'Sepal length', 'Ligule width', 'Style length', 'Ovule number']  # Adjusted based on your column names

    # Separate data into two parts: one with missing values and one without
    data_missing = result[result[features_with_missing].isnull().any(axis=1)]
    data_complete = result.dropna(subset=features_with_missing)

    # Convert object columns to numeric if needed
    # Convert object columns to categorical variables
    object_columns = data_complete.select_dtypes(include=['object']).columns
   # data_complete[object_columns] = data_complete[object_columns].astype('category')
    data_complete.loc[:, object_columns] = data_complete.loc[:, object_columns].astype('category')


    # Perform one-hot encoding for categorical variables
    data_complete = pd.get_dummies(data_complete)

    # Split the complete data into features and target after encoding
    X = data_complete.drop(features_with_missing, axis=1)
    # Check if columns exist before dropping
    columns_to_drop = [col for col in features_with_missing if col in data_complete.columns]
    X = data_complete.drop(columns_to_drop, axis=1)
    # Specify the column order when dropping features for prediction
    columns_to_drop = [col for col in features_with_missing if col in data_complete.columns]
    X_missing = data_missing.drop(columns_to_drop, axis=1)[X.columns]

    # Check if columns exist before accessing
    columns_to_access = [col for col in features_with_missing if col in data_complete.columns]
    y = data_complete[columns_to_access]


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model for each feature with missing values
    imputed_values = {}
   # Predict missing values using the trained models
    imputed_values = {}
    for feature in features_with_missing:
        if feature in y_train.columns and feature in y_test.columns:
            model = RandomForestRegressor()
            model.fit(X_train, y_train[feature])
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test[feature], y_pred)
            print(f"Mean Squared Error for {feature}: {mse}")
            imputed_values[feature] = model.predict(X_missing)
        else:
            print(f"Feature {feature} not found in y_train or y_test.")


    # Impute missing values in the original dataset
    for feature in features_with_missing:
        result.loc[result[feature].isnull(), feature] = imputed_values[feature]
    # Now 'result' contains the imputed values
        
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