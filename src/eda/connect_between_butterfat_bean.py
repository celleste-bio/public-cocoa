
"""
left join on the butterfat and bean tables

"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd


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

def left_join(connection):
    # Define your SQL query with LEFT JOIN
    query_join_tables = """
    SELECT *
    FROM butterfat
    LEFT JOIN bean ON butterfat.[Clone name] = bean.[Clone name] AND butterfat.[Refcode] = bean.[Refcode]
    WHERE butterfat.fat IS NOT NULL
    """
    # Use read_sql_query with the connection 
    result = pd.read_sql_query(query_join_tables, connection)
    result.info()
    result.head()

    


def main():
    #script_path="/home/public-cocoa/src/eda/connect_between_butterfat_bean.py"
    #create database
    script_path = os.path.realpath(__file__)
    db_path=create_path(script_path)
    #Create a SQLite database connection
    connection = sqlite.connect(db_path)

    connection.close()

if __name__ == "__main__":
    main()