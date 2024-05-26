'''
Extracting nacodes for clone name search
'''

# dependencies
import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm
from urllib.parse import urlparse, parse_qs
import json
import re

sys.path.append("/home/public-cocoa/src/")
from path_utils import go_back_dir

def get_all_links(page):
    links = page.find_all('a')
    hrefs = [link.get('href') for link in links]
    return hrefs

def clean_text(text):
    cleaned_text = text.replace('\u00a0', ' ')
    cleaned_text = text.replace('Show Details Hide Details ', '')

    # Remove extra whitespaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def get_titles(title_parts):
    titles = []
    for title_part in title_parts:
        if title_part and isinstance(title_part, Tag):
            title = title_part.find("a", {"class": "mainlink"})
            if title:
                title_text = title.text.replace('\xa0', '').lower()
                titles.append(title_text)
    return titles

def get_clone_pedigree(nacode):
    url = f"https://www.icgd.reading.ac.uk/all_data.php?nacode={nacode}"
    response = requests.get(url)
    response_content = response.content
    page = BeautifulSoup(response_content, "html.parser")
    data_section = page.find("div", {"id": "accordion"})
    
    clone_connection = set()
    if data_section:
        title_parts = data_section.find_all("h4")
        titles = get_titles(data_section)
        if "pedigree details" in titles:
            pedigree_index = titles.index("pedigree details")
            if pedigree_index is not None: 
                pedigree_part = title_parts[pedigree_index]
                clone_links = pedigree_part.find_all_next("a", {"class" : "clonelink"})
                for link in clone_links:
                    parsed_data_link = urlparse(link["href"])
                    nacode = parse_qs(parsed_data_link.query).get("nacode", [])[0]
                    clone_connection.add(nacode)
            
    return list(clone_connection)

def scrape_info(collection_file):
    collection = pd.read_csv(collection_file, dtype=str)
    result_dict = {}
    
    for index, row in tqdm(collection.iterrows(), desc="Getting Pedigree"):
        nacode = row["NA Code"]
        # clone_name = row["Clone Name"]
        result = get_clone_pedigree(nacode)
        if result:
            result_dict[nacode] = result

    return result_dict

def save_as_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 2)
    collection_file = os.path.join(project_path, "data", "cocoa_collection.csv")
    pedigree_result = scrape_info(collection_file)

    output_file = os.path.join(project_path, "data", "pedigree.json")
    save_as_json(pedigree_result, output_file)