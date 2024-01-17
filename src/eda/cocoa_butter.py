"""
exploring butter fat related data
"""

# packages
import os
import sqlite3 as sqlite
import pandas as pd
import matplotlib.pyplot as plt

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

    # data
    df = pd.read_sql_query("SELECT * FROM butterfat;", connection)
    df.info()
    fat_info = df[["Clone name", "Refcode", "Fat"]].dropna()
    fat_info["Fat"] = pd.to_numeric(fat_info["Fat"], errors='coerce')
    fat_info = fat_info.dropna()

    # plot
    fat_info["Fat"].plot(kind='density', color='blue', linewidth=2)
    plt.xlabel('Fat %')
    plt.ylabel('Density')
    plt.title('Fat precentage distribution')
    plt.show()

    connection.close()

if __name__ == "__main__":
    main()

