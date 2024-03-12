'''
Main script to build database from public data
'''

# dependencies
import os
from path_utils import go_back_dir
from download_icgd_collection import save_collection
from update_nacode import update_nacodes
from scrape_ref_links import scrape_ref_links
from download_ref_tables import download_tables

from combine_tables import extract_table_info, add_refcode_column, combine_tables
from build_sqlite import build_database

def get_data(project_path):
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

def sort_data(project_path):
    tables_dir = os.path.join(project_path, "data", "tables")
    all_files = os.listdir(tables_dir)

    combined_tables_dir = os.path.join(project_path, "data", "combined_tables")
    if not os.path.exists(combined_tables_dir):
        os.makedirs(combined_tables_dir)

    print("combinning tables")
    tables_by_subject = {}

    for file_name in all_files:
        refcode, subject = extract_table_info(file_name)
        file_path = os.path.join(tables_dir, file_name)
        df = add_refcode_column(file_path, refcode)
        if subject not in tables_by_subject:
            tables_by_subject[subject] = []
        tables_by_subject[subject].append(df)

    for subject, tables in tables_by_subject.items():
        combined_df = combine_tables(tables)
        new_file_path = os.path.join(combined_tables_dir, f"{subject}.csv")
        combined_df.to_csv(new_file_path, index=False)

def main():
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 2)

    get_data(project_path)

    sort_data(project_path)

    build_database(project_path)


if __name__ == "__main__":
    main()