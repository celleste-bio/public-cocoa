"""
connection between:
* geo group -> growing countries
* geno group -> cocoa variety
* pheno group -> taste ranking
"""

# packages
import os
import sys
import sqlite3 as sqlite
import pandas as pd

sys.path.append("/home/public-cocoa/src/")
from path_utils import go_back_dir

import statsmodels.api as sm
from statsmodels.formula.api import ols

def main():
    # context
    script_path = os.path.realpath(__file__)
    project_path = go_back_dir(script_path, 2)
    db_path = os.path.join(project_path, "data", "ICGD.db")
    connection = sqlite.connect(db_path)
    
    query = "SELECT rating, bean_origin FROM samples"
    taste_info = pd.read_sql_query(query, connection)
    
    # Perform ANOVA to calculate the variabily explaind by origin country
    # Formula: 'target ~ C(feature)' specifies that 'target' is the response variable and 'feature' is the categorical predictor
    model = ols('rating ~ C(bean_origin)', data=taste_info).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    mean_ratings = taste_info.groupby('bean_origin')['rating'].mean().reset_index()
    mean_ratings.columns = ['bean_origin', 'mean_rating']

    ref_origin_path = os.path.join(project_path, "data", "ref_origin.csv")
    ref_origin = pd.read_csv(ref_origin_path)
    ref_origin = ref_origin[["Refcode", "Country"]]

    query = "SELECT [Clone name], [Refcode] FROM bean"
    bean_info = pd.read_sql_query(query, connection)

    query = "SELECT [Clone name], [Population] FROM groups"
    groups_info = pd.read_sql_query(query, connection)

    merged_data = pd.merge(bean_info, ref_origin, on='Refcode', how='left')
    merged_data = pd.merge(merged_data, groups_info, on='Clone name', how='left')
    merged_data[["Clone name", "Country"]]


    # Group by "Country" and "Clone name" to count occurrences
    count_data = merged_data.groupby(['Country', 'Clone name']).size().reset_index(name='Count')

    # Group by "Country" to get the total number of rows for each country
    total_counts = merged_data.groupby('Country').size().reset_index(name='Total')

    # Merge the count data with the total counts
    frequency_data = pd.merge(count_data, total_counts, on='Country')

    # Calculate the frequency
    frequency_data['Frequency'] = frequency_data['Count'] / frequency_data['Total']

    # Drop the Count and Total columns if you only want to keep the Frequency
    frequency_data = frequency_data.drop(columns=['Count', 'Total'])

    # Merge frequency data with mean ratings
    merged = pd.merge(frequency_data, mean_ratings, left_on='Country', right_on='bean_origin')

    # Calculate the weighted rating
    merged['Weighted Rating'] = merged['Frequency'] * merged['mean_rating']

    # Group by clone name to sum up the weighted ratings
    clone_ratings = merged.groupby('Clone name')['Weighted Rating'].sum().reset_index()
    sorted_clone_ratings = clone_ratings.sort_values(by='Weighted Rating', ascending=False).reset_index(drop=True)
    sorted_clone_ratings.head(n = 10)
    merged_data = pd.merge(sorted_clone_ratings, groups_info, on='Clone name', how='left')
    merged_data.to_csv( os.path.join(project_path, "data", "clone_ratings.csv"), index=False)

    connection.close()

if __name__ == "__main__":
    main()
