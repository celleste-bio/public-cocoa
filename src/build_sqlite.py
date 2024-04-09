'''
Building sqlite database
'''

# dependencies
import os
import sqlite3
import pandas as pd

def import_csv_to_sqlite(csv_file, table_name, connection):
    try:
        df = pd.read_csv(csv_file)
        if not df.empty:
            df.to_sql(table_name, connection, index=False, if_exists="replace")
        else:
            print(f"Warning: DataFrame from {csv_file} is empty. Skipping table creation.")
    except pd.errors.EmptyDataError:
        print(f"Error: No data to parse from {csv_file}.")

def import_dir_to_sqlite(dir_path, connection):
    table_files = os.listdir(dir_path)
    table_names = [table.split('.')[0] for table in table_files]

    tables_dict = {table_name : os.path.join(dir_path, csv_file) for csv_file, table_name in zip(table_files, table_names)}
    for table_name, csv_file in tables_dict.items():
        import_csv_to_sqlite(csv_file, table_name, connection)


def build_database(project_path):
    database_name = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite3.connect(database_name)
    ref_info_file = os.path.join(project_path, "data", "ref_info.csv")
    import_csv_to_sqlite(ref_info_file, "ref_info", connection)

    flavors_dir = os.path.join(project_path, "data", "flavors")
    import_dir_to_sqlite(flavors_dir, connection)

    combined_tables_dir = os.path.join(project_path, "data", "combined_tables")
    import_dir_to_sqlite(combined_tables_dir, connection)

    connection.close()
