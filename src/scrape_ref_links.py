'''
Extracting reference links for all table info
'''

# dependencies
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm


def get_all_links(page):
    links = page.find_all("a")
    hrefs = [link.get("href") for link in links]
    return hrefs

def filter_for_ref_links(links):
    base_url = "https://www.icgd.reading.ac.uk/"
    ref_links = [base_url+link for link in links if "ref_data.php?refcode=" in link]
    return ref_links

def get_clone_ref_links(nacode):
    url = f"https://www.icgd.reading.ac.uk/all_data.php?nacode={nacode}"
    response = requests.get(url)
    response_content = response.content
    page = BeautifulSoup(response_content, "html.parser")
    links = get_all_links(page)
    ref_links = filter_for_ref_links(links)
    return ref_links

def scrape_ref_links(collection_file):
    collection = pd.read_csv(collection_file, dtype=str)
    result = set()

    # Add a progress bar to the loop
    for index, row in tqdm(collection.iterrows(), total=len(collection), desc="Scraping Ref Links"):
        nacode = row["NA Code"]
        ref_links = get_clone_ref_links(nacode)
        result.update(ref_links)

    return result

# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    collection_file = os.path.join(project_path, "data", "cocoa_collection.csv")
    result = scrape_ref_links(collection_file)

    output_file = os.path.join(project_path, "data", "ref_links.txt")
    with open(output_file, 'w') as file:
        for element in result:
            file.write(str(element) + "\n")
