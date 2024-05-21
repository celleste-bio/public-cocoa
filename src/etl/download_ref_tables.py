'''
downloads all tables from reference links
'''

# dependencies
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import re

def get_url_params(url):
    parsed_url = urlparse(url)
    return parse_qs(parsed_url.query)

def get_param_value(params, param_name, default=''):
    return params.get(param_name, [default])[0]

def get_refcode(url):
    params = get_url_params(url)
    return get_param_value(params, 'refcode')

def generate_file_name(url):
    params = get_url_params(url)
    refcode = get_param_value(params, 'refcode')
    table = get_param_value(params, 'table')
    return f"{refcode}_{table}.csv"

def download_tables(ref_links_file, output_dir):
    data_dir = os.path.dirname(output_dir)
    download_ref_info(ref_links_file, data_dir)
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

def extract_year(description):
    pattern = r'\((\d{4})\)'
    matches = re.findall(pattern, description)
    if matches:
        year = matches[0]
    else:
        year = ''
    return year

def download_ref_info(ref_links_file, output_dir):
    with open(ref_links_file, 'r') as file:
        ref_links = file.readlines()

    ref_info = pd.DataFrame(columns=["refcode", "title", "year"])
    refcodes = set()
    for link in tqdm(ref_links, desc="Downloading References Info"):
        refcode = get_refcode(link)
        if refcode not in refcodes:
            refcodes.update([refcode])
            response = requests.get(link)
            if response.status_code == 200:
                page = BeautifulSoup(response.content, "html.parser")
                description = page.find("p")
                title = description.text
                year = extract_year(title)
                new_row = {"refcode": refcode, "title": title, "year": year}
                ref_info = ref_info._append(new_row, ignore_index=True)

    output_file = os.path.join(output_dir, "ref_info.csv")
    ref_info.to_csv(output_file, index=False)
         
# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    ref_links_file = os.path.join(project_path, "data", "ref_links.txt")

    data_dir = os.path.join(project_path, "data")
    tables_dir = os.path.join(project_path, "data", "tables")
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    download_tables(ref_links_file, tables_dir)
