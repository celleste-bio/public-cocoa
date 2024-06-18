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

def fill_missing_values(df):
    # Process float64 columns
    df=convert_to_float(df)
    for column in df.select_dtypes(include=['float64']).columns:
        mean = df[column].mean()
        std = df[column].std()

        # Count of missing values in the column
        nan_count = df[column].isna().sum()

        # Generate random values from normal distribution with calculated mean and std
        random_values = np.random.normal(loc=mean, scale=std, size=nan_count)

        # Fill the NaN values with these random values
        df.loc[df[column].isna(), column] = random_values

    # Process object columns
    for column in df.select_dtypes(include=['object']).columns:
        mode = df[column].mode()[0]  # Most frequent value (mode)

        # Fill the NaN values with the mode
        df[column].fillna(mode, inplace=True)

    return df


if __name__ == "__main__":
    fill_missing_values()