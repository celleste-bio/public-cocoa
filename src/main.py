'''
Main script to build database from public data
'''

# dependencies
import os
import json
from download_icgd_collection import save_collection
from update_nacode import update_nacodes
from scrape_clone_info import scrape_info

def save_as_json(data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def main():
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    collection_file = os.path.join(project_path, "data", "cocoa_collection.csv")

    # agreggates the collection into a single file in data/cocoa_collection.csv
    save_collection(collection_file)
    # retrives nacodes based on clonename search
    update_nacodes(collection_file)
    # retrives all the data based on the collection file, saved as data.json
    result = scrape_info(collection_file)

    output_file = os.path.join(project_path, "data", "data.json")
    save_as_json(result, output_file)

if __name__ == "__main__":
    main()