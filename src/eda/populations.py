"""
exploring populations data
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

def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 3)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    
    # fat data
    pop_info = pd.read_sql_query("SELECT * FROM groups", connection)
    pop_info["Population"].value_counts()
    pop_info[["Clone name", "Population", "Refcode"]]

    connection.close()

if __name__ == "__main__":
    main()

