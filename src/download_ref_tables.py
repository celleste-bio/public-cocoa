'''
downloads all tables from reference links
'''

# dependencies
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm


def generate_file_name(url):
    parsed_url = urlparse(url)
    params = parse_qs(parsed_url.query)
    refcode = params.get('refcode', [''])[0]
    table = params.get('table', [''])[0]
    file_name = f"{refcode}_{table}.csv"
    return file_name

def download_tables(ref_links_file, output_dir):
    with open(ref_links_file, 'r') as file:
        ref_links = file.readlines()

    for link in tqdm(ref_links, desc="Downloading Tables"):
        try:
            table = pd.read_html(link)[0]
            output_file_name = generate_file_name(link)
            output_file = os.path.join(output_dir, output_file_name)
            table.to_csv(output_file, index=False)
        except ValueError as ve:
            print("Error loading DataFrame from:", link)
         
# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    ref_links_file = os.path.join(project_path, "data", "ref_links.txt")

    tables_dir = os.path.join(project_path, "data", "tables")
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    download_tables(ref_links_file, tables_dir)
