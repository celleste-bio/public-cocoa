"""
exploring the effects of breeding
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

    ############################################################
    
    # fat data
    fat_info = pd.read_sql_query(
        """
        SELECT *
        FROM butterfat
        JOIN ref_info ON butterfat.Refcode = ref_info.refcode;
        """,
        connection
    )
    fat_info = fat_info[["Clone name", "Refcode", "Fat", "year"]].dropna()
    fat_info["Fat"] = pd.to_numeric(fat_info["Fat"], errors='coerce')
    fat_info = fat_info.dropna()

    grouped_data = fat_info.groupby('year')['Fat'].agg(['mean', 'std'])

    plt.plot(grouped_data.index, grouped_data['mean'], label='Mean Line', color='red')

    plt.fill_between(
        grouped_data.index,
        grouped_data['mean'] - grouped_data['std'],
        grouped_data['mean'] + grouped_data['std'],
        facecolor='blue', alpha=0.3, label='Std Dev Ribbon'
    )

    plt.xlabel('Year')
    plt.ylabel('Fat Precentage')
    plt.title('Mean Fat Precentage Over Years with Standard Deviation')
    plt.legend()
    plt.grid(False)
    plt.show()

    ############################################################

    # yield data
    yield_info = pd.read_sql_query(
        """
        SELECT *
        FROM yield
        JOIN ref_info ON yield.Refcode = ref_info.refcode;
        """,
        connection
    )
    yield_info = yield_info[["Clone name", "Refcode", "Yield ha", "year"]].dropna()
    yield_info["Yield ha"] = pd.to_numeric(yield_info["Yield ha"], errors='coerce')
    yield_info = yield_info.dropna()

    grouped_data = yield_info.groupby('year')['Yield ha'].agg(['mean', 'std'])

    plt.plot(grouped_data.index, grouped_data['mean'], label='Mean Line', color='red')

    plt.fill_between(
        grouped_data.index,
        grouped_data['mean'] - grouped_data['std'],
        grouped_data['mean'] + grouped_data['std'],
        facecolor='blue', alpha=0.3, label='Std Dev Ribbon'
    )

    plt.xlabel('Year')
    plt.ylabel('Yield ha')
    plt.title('Mean Yield ha Over Years with Standard Deviation')
    plt.legend()
    plt.grid(False)
    plt.show()

    ############################################################

    connection.close()

if __name__ == "__main__":
    main()

