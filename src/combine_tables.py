'''
Combines all tables by category
'''

# dependencies
import os
import pandas as pd

def extract_table_names(all_tables):
    table_names = set()
    for table in all_tables:
        # Extract words between '_' and '.csv'
        words = table.split('_')
        if len(words) > 1:
            table_name = words[1].split('.')[0]
            table_names.add(table_name)
    return table_names

def combine_tables(tables_files):
    # Combine tables into a single DataFrame
    combined_df = pd.DataFrame()
    for file in tables_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    return combined_df