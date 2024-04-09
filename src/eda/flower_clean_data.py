
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
import seaborn as sns
from scipy.stats import norm,f_oneway
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy import stats


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

    #replacing "-" with NaN in both numerical and categorical columns
    
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

    #dealing with outliers
    
    columns_numbers=["Sepal length", "Ligule width", "Ovule number","Style length"]
    df = result
    df[columns_numbers] = df[columns_numbers].astype(float)
    Q1= df[columns_numbers].quantile(0.25)
    Q3= df[columns_numbers].quantile(0.75)
    # Calculate IQR
    IQR = Q3 - Q1
    # Define boundaries for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Replace outliers with median
    for column in df.columns:
      if df[column].dtype=='float64' or df[column].dtype=='int64':
        median = df[column].median()
        df[column] = np.where((df[column] < lower_bound[column]) | (df[column] > upper_bound[column]), median, df[column])
        df[column].fillna(median,inplace=True)

    df.boxplot()
    plt.show()

    # Load your dataset
    result.info()
        
    result.head()
     #understanding the column.
    for column in columns_numbers:
        plt.figure(figsize=(8, 6))  # Set the figure size (optional)
        precent=df[column].value_counts()/df[column].value_counts().sum()
        precent.plot(kind='bar')
        for index, value in enumerate(precent):
            plt.text(index, value, str(round(value, 2)), ha='center', va='bottom')
        plt.title('Probability Graph for {} based on the number of existing values in the column'.format(column))
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent clipping of annotations
        plt.show()

    copy_data_numeric=df.copy()[df.select_dtypes(include='float64').columns]
    #fill missing values category and floating numbers:
    for col in copy_data_numeric.columns:
        mean_val = copy_data_numeric[col].median()
        copy_data_numeric[col].fillna(mean_val, inplace=True)

    # Step 2: Perform K-means clustering k=10
    inertia_=[]
    for k in range(1,11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(copy_data_numeric)
        inertia_.append(kmeans.inertia_)

    #step 3 choosing the k
    plt.plot(range(1,11),inertia_)
    plt.show()

    float_columns=df.select_dtypes(include='float64').columns
    category_columns=df.select_dtypes(include='object').columns
    category_to_replace=category_columns
    copy_data=df.copy()
    #fill missing values of floating numbers
    for col in float_columns:
        mean_val = copy_data[col].median()
        copy_data[col].fillna(mean_val, inplace=True)
  # Fill missing values of categorical columns with most frequent value
    for col in category_columns:
        frequent_val = copy_data[col].mode()[0]
        copy_data[col].fillna(frequent_val, inplace=True)
    sc_data=MinMaxScaler()
    sc_data=sc_data.fit_transform(copy_data[float_columns])
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sc_data)
    clusters= kmeans.labels_

# Step 3: Calculate cluster means
    cluster_means = []
    for i in range(k):
        cluster_data = copy_data[float_columns][clusters == i]
        cluster_mean = np.mean(cluster_data, axis=0)
        cluster_means.append(cluster_mean)

 #Calculate frequent values for categorical columns within each cluster
    cluster_frequent_values_categorical = []
    for i in range(k):
        cluster_data = copy_data[category_to_replace][clusters == i]
        frequent_values = cluster_data.mode().iloc[0]
        cluster_frequent_values_categorical.append(frequent_values)



# Step 4: Replace missing values with cluster means
    for index in copy_data.index:
        cluster_idx = clusters[index]
    for column in float_columns:
        if  pd.isnull(df.loc[index,column]):
            cluster_mean = cluster_means[cluster_idx]
            df.loc[index, column] = cluster_mean[column]

    for col in category_to_replace:
        if df.loc[index, col]=='-':
            cluster_frequent_val = cluster_frequent_values_categorical[cluster_idx][col]
            df.loc[index, col] = cluster_frequent_val



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