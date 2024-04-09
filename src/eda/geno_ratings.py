"""
exploring genotype data and cocoa ratings
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
from ..path_utils import go_back_dir

def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    
    query = """
        SELECT
            bean_origin, rating
        FROM
            samples
    """

    data = pd.read_sql_query(query, connection)
    origin_ratings = data.groupby('bean_origin')['rating'].mean().reset_index()

    ref_origin_path = os.path.join(project_path, "ref_origin.csv")
    ref_origin = pd.read_csv(ref_origin_path)

    cocoa_collection_path = os.path.join(project_path, "data","cocoa_collection.csv")
    cocoa_collection = pd.read_csv(cocoa_collection_path)