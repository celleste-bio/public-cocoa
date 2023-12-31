'''
Downloads the latest International Cocoa Germplasm Database (ICGD) collection info
'''

# dependencies
import os
import requests
import pandas as pd
from io import StringIO

def get_df_from_link(url):

    response = requests.get(url)
    if response.status_code == 200:
        csv_content = response.text
        df = pd.read_csv(StringIO(csv_content), skiprows=[0,1])
        return df
    else:
        print(f"Failed to retrieve CSV file. Status code: {response.status_code}")

def save_collection(collection_file):

    collection_link = "https://www.icgd.reading.ac.uk/icqc/list_output.php?list=post"
    quarantine_link = "https://www.icgd.reading.ac.uk/icqc/list_output.php?list=quar"

    collection = get_df_from_link(collection_link)
    quarantine = get_df_from_link(quarantine_link)

    full_collection = pd.concat([collection, quarantine], ignore_index=True)
    full_collection.to_csv(collection_file, index=False)

# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    collection_file = os.path.join(project_path, "data", "cocoa_collection.csv")
    save_collection(collection_file)