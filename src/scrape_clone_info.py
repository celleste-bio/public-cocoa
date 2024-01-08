'''
Extracting nacodes for clone name search
'''

# dependencies
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import json
import re

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

def get_clone_data(nacode):
    url = f"https://www.icgd.reading.ac.uk/all_data.php?nacode={nacode}"
    response = requests.get(url)
    response_content = response.content
    page = BeautifulSoup(response_content, "html.parser")
    data_section = page.find("div", {"id": "accordion"})

    # Check if data_section is found
    if data_section:
        title_parts = data_section.find_all("h4")

        data_dict = {}
        for title_part in title_parts:
            title = title_part.find("a", {"class": "mainlink"})
            if title:
                title_text = title.text.replace('\xa0', '').lower()
                data_div = title_part.find_next("div")
                data = clean_text(data_div.text)
                data_dict[title_text] = data

        return data_dict
    else:
        # Handle the case where 'accordion' is not found
        return {}

def scrape_info(collection_file):
    collection = pd.read_csv(collection_file, dtype=str)
    result_dict = {}
    
    for index, row in collection.iterrows():
        nacode = row["NA Code"]
        clone_name = row["Clone Name"]
        result = get_clone_data(nacode)
        if result:
            result_dict[clone_name] = result

    return result_dict

def save_as_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    collection_file = os.path.join(project_path, "data", "cocoa_collection.csv")
    result = scrape_info(collection_file)

    output_file = os.path.join(project_path, "data", "data.json")
    save_as_json(result, output_file)