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

def Filling_missing_values_by_IQR(df,columns_numbers):
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
        df[column] = np.where((df[column] < lower_bound[column]) | (df[column] > upper_bound[column]), np.nan, df[column])

    return df

def IQR_outliers(df):
    df = convert_to_float(df)
    numeric_columns = [column for column in df.columns if df[column].dtype=='int64' or df[column].dtype=='float64']
    df = Filling_missing_values_by_IQR(df,numeric_columns)
    return df