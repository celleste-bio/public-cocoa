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


def fill_missing_values_median_and_mode(df):
    df=convert_to_float(df)
    # Process float64 columns
    for column in df.select_dtypes(include=['float64']).columns:
        median = df[column].median()

        # Fill the NaN values with the median
        df[column].fillna(median, inplace=True)

    # Process object columns
    for column in df.select_dtypes(include=['object']).columns:
        mode = df[column].mode()[0]  # Most frequent value (mode)

        # Fill the NaN values with the mode
        df[column].fillna(mode, inplace=True)

    return df

if __name__ == "__main__":
    fill_missing_values_median_and_mode()