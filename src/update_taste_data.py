import sqlite3
import csv
from build_sqlite import import_csv_to_sqlite
import os
import pandas as pd

script_path="/home/public-cocoa/src/update_taste_data.py"
database_name = os.path.dirname(os.path.dirname(script_path))
data_path="/home/public-cocoa/cocoa_ratings.tsv"
csv_file= pd.read_csv(data_path,delimiter="\t")
tables= ["Companies","References","Tastes", "Samples"]
for table in tables:
    connection = sqlite3.connect(csv_file,table,database_name,)

