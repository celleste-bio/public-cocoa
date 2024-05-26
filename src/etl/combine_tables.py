'''
Combines all tables by category
'''

# dependencies
import os
import pandas as pd

def extract_table_info(file_name):
    """
    Extracts refcode and subject from file name
    """
    words = file_name.split('_')
    if len(words) > 1:
        refcode = words[0]
        subject = words[1].split('.')[0]
    return refcode, subject

def add_refcode_column(file_path, refcode):
    df = pd.DataFrame()
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["Refcode"] = refcode
    else:
        print("File not exists")
    return df

def combine_tables(tables):
    """
    Combine tables into a single DataFrame
    """
    combined_df = pd.DataFrame()
    for table in tables:
        combined_df = pd.concat([combined_df, table], ignore_index=True)
    return combined_df