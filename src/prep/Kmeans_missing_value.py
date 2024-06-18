from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def choosing_the_K(data):
  data = convert_to_float(data)
  copy_data_numeric=data.copy()[data.select_dtypes(include='float64').columns]
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

def clustering_and_replace_missing_values(df,k):
    copy_data=df
    float_columns=copy_data.select_dtypes(include='float64').columns
    category_columns=copy_data.select_dtypes(include='object').columns

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
        cluster_data = copy_data[category_columns][clusters == i]
        frequent_values = cluster_data.mode().iloc[0]
        cluster_frequent_values_categorical.append(frequent_values)

    copy_data = copy_data.reset_index(drop=True)
    # Adjust the clusters dictionary if necessary
    copy_data.info()
    # Step 4: Replace missing values with cluster means
    for index in copy_data.index:
        cluster_idx = clusters[index]
        for column in float_columns:
            if  pd.isnull(copy_data.loc[index,column]):
                cluster_mean = cluster_means[cluster_idx]
                copy_data.loc[index, column] = cluster_mean[column]
        for col in category_columns:
            if copy_data.loc[index, col]=='-' or pd.isnull(copy_data.loc[index,col]) :
                cluster_frequent_val = cluster_frequent_values_categorical[cluster_idx][col]
                copy_data.loc[index, col] = cluster_frequent_val
    return df

