'''
Downloads cocoa ratings data from https://flavorsofcacao.com/chocolate_database.html
'''

# dependencies
import os
from bs4 import BeautifulSoup
import requests
import pandas as pd

def save_flavors_data(collection_file):
    data_source = "https://flavorsofcacao.com/chocolate_database.html"
    response = requests.get(data_source)
    html_content = response.text
    soup = BeautifulSoup(html_content, 'html.parser')
    table_part = soup.find('div', id='spryregion1')
    df = pd.read_html(data_source)
    

# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    output_file = os.path.join(project_path, "data", "cocoa_ratings.csv")
    save_flavors_data(output_file)

