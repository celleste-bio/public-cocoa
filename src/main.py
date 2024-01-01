'''
Main script to build database from public data
'''

# dependencies
import os
from download_icgd_collection import save_collection
from update_nacode import update_nacodes
from scrape_ref_links import scrape_ref_links
from download_ref_tables import download_tables

def main():
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))

    data_dir = os.path.join(project_path, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    collection_file = os.path.join(project_path, "data", "cocoa_collection.csv")

    print("getting collection")
    # agreggates the collection into a single file in data/cocoa_collection.csv
    save_collection(collection_file)

    print("updating nacodes")
    # retrives nacodes based on clonename search
    update_nacodes(collection_file)

    print("extracting reference links")
    ref_links = scrape_ref_links(collection_file)
    ref_links_file = os.path.join(project_path, "data", "ref_links.txt")
    with open(ref_links_file, 'w') as file:
        for link in ref_links:
            file.write(str(link) + "\n")

    tables_dir = os.path.join(project_path, "data", "tables")
    if not os.path.exists(tables_dir):
        os.makedirs(tables_dir)

    print("downloading tables")
    download_tables(ref_links_file, tables_dir)

if __name__ == "__main__":
    main()