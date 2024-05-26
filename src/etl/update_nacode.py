'''
Extracting nacodes for clone name search
'''

# dependencies
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm

def search_nacode(name):
    nacode = ""
    replacements = {" ": "+", "/": "%2F", "[": "%5B", "]": "%5D"}
    name_url = name.translate(str.maketrans(replacements))
    # name_url = name.replace(" ", "+").replace("/", "%2F").replace("[", "%5B").replace("]", "%5D")
    url = f"https://www.icgd.reading.ac.uk/search_name.php?clonename={name_url}"
    response = requests.get(url)
    response_content = response.content
    page = BeautifulSoup(response_content, "html.parser")
    data_button = page.find("a", {"class": "btn-yellow"})
    if data_button and ("href" in data_button.attrs):
        data_link = data_button["href"]
        parsed_data_link = urlparse(data_link)
        nacode = parse_qs(parsed_data_link.query).get("nacode", [])[0]
    return nacode

def update_nacodes(collection_file):
    collection = pd.read_csv(collection_file)
    collection["NA Code"] = ''

    # Add a progress bar to the loop
    for index, name in tqdm(enumerate(collection["Clone Name"]), total=len(collection), desc="Updating NA Codes"):
        nacode = search_nacode(name)
        collection.loc[index, "NA Code"] = nacode

    collection.to_csv(collection_file, index=False)

# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    collection_file = os.path.join(project_path, "data", "cocoa_collection.csv")
    update_nacodes(collection_file)