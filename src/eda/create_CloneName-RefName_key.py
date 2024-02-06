# connect between the tables

import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import numpy as np
import os

def go_back_dir(path, number):
    result = path
    for i in range(number):
        result = os.path.dirname(result)
    return result

def create_path(__file__):
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    return(db_path)

# Uploading the CSV File of 'yield' and learning about it
bf_df = pd.read_csv('/content/yield.csv')
bf_df.info()
bf_df.head()

# Step 1: Create a new column by concatenating "Clone name" and "Refcode"
bf_df['Clone name + Refcode'] = bf_df['Clone name'].astype(str) + '_' + bf_df['Refcode'].astype(str)

# Step 2: Drop the "Clone name" and "Refcode" columns
bf_df.drop(['Clone name', 'Refcode'], axis=1, inplace=True)

# Display the updated DataFrame
bf_df.info()
bf_df.head()

# Check the number of unique values in the "NewColumn"
num_unique_values = bf_df['Clone name + Refcode'].nunique()

# Display the result
print(f'The number of unique values in the "NewColumn" is: {num_unique_values}')

# Specify the path where you want to save the CSV file
csv_path = '/content/yield_v1.csv'

# Export the DataFrame to CSV
bf_df.to_csv(csv_path, index=False)

# Print a message indicating the successful export
print(f'DataFrame has been exported to {csv_path}')

# Specify the path where you want to save the CSV file
csv_path = '/content/yield_v1.csv'

# Export the DataFrame to CSV
bf_df.to_csv(csv_path, index=False)

# Print a message indicating the successful export
print(f'DataFrame has been exported to {csv_path}')

