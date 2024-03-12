"""
Exploratory Data Analysis - Workshop
"""

# packages
import os
import pandas as pd
from scipy.stats import ttest_ind

script_path = __file__
project_path = os.path.dirname(os.path.dirname(script_path))

data_source = "https://flavorsofcacao.com/chocolate_database.html"
data_path = os.path.join(project_path, "data", "cocoa_ratings.tsv")
data = pd.read_csv(data_path, delimiter = "\t")


data.head(10)
data.info()
column_names = data.columns
column_names = [
    'REF', 'Company (Manufacturer)', 'Company Location', 'Review Date',
    'Country of Bean Origin', 'Specific Bean Origin or Bar Name',
    'Cocoa Percent', 'Ingredients', 'Most Memorable Characteristics',
    'Rating'
]

categorical_nominal = ['REF', 'Company (Manufacturer)', 'Company Location', 'Country of Bean Origin', 'Specific Bean Origin or Bar Name', 'Ingredients', 'Most Memorable Characteristics']
categorical_ordinal = []
numerical_discrete = ['Review Date', 'Rating']
numerical_continuous = ['Cocoa Percent']

multivalue_columns = ['Ingredients', 'Most Memorable Characteristics']
data["sample_id"] = data.index

references = data[['REF', 'Review Date']] \
    .rename(columns={'REF': 'ref_code', 'Review Date': 'year'}) \
    .drop_duplicates()

companies = data[['Company (Manufacturer)', 'Company Location']] \
    .rename(columns={'Company (Manufacturer)': 'name', 'Company Location': 'location'}) \
    .drop_duplicates()

samples = data[['sample_id', 'REF', 'Company (Manufacturer)', 'Country of Bean Origin', 'Specific Bean Origin or Bar Name', 'Ingredients', 'Cocoa Percent', 'Rating']] \
    .rename(columns={'Country of Bean Origin': 'bean_origin', 'REF': 'ref_code', 'Company (Manufacturer)': 'company', 'Specific Bean Origin or Bar Name': 'bar_name', 'Ingredients': 'ingredients', 'Cocoa Percent': 'cocoa_percent', 'Rating': 'rating'}) \
    .drop_duplicates()

tastes = data[['sample_id', 'Most Memorable Characteristics']].drop_duplicates() \
    .rename(columns={'Most Memorable Characteristics': 'characteristics'}) \
    .drop_duplicates()

characteristics = set()
for taste_list in tastes['characteristics']:
    for taste in taste_list.split(','):
        if taste.strip():
            characteristics.add(taste.strip())

def split_ingredients(ingredient_str):
    num_ingr, ingr_str = ingredient_str.split('-')
    ingr_codes = ingr_str.split(',')
    return ingr_codes

def decode_ingredients(ingredient_code):
    ingr_index = {
        'B' : 'beans',
        'S' : 'sugar',
        'S*' : 'sweetener',
        'C' : 'cocoa butter',
        'V' : 'vanilla',
        'L' : 'lecithin',
        'Sa' : 'salt'
    }
    ingredient = ingr_index.get(ingredient_code)
    return ingredient

def get_ingredient_dict(ingredients):
    col_names = ['beans', 'sugar', 'sweetener', 'cocoa butter', 'vanilla', 'lecithin', 'salt']
    results = {col : (col in ingredients) for col in col_names}
    return results

def break_ingredients(ingredient_column):
    ingredient_names = ['beans', 'sugar', 'sweetener', 'cocoa butter', 'vanilla', 'lecithin', 'salt']
    ingredients_df = pd.DataFrame(columns=ingredient_names)
    
    for ingredient_str in ingredient_column:
        ingredient_codes = split_ingredients(ingredient_str)
        ingredients = [decode_ingredients(code.strip()) for code in ingredient_codes]
        new_row = pd.DataFrame([get_ingredient_dict(ingredients)])
        ingredients_df = pd.concat([ingredients_df, new_row], axis=0)
    
    return ingredients_df

most_common_ingredients = samples["ingredients"].value_counts().reset_index().loc[0, "ingredients"]
samples.replace({' ': most_common_ingredients}, inplace=True)

ingredients_df = break_ingredients(samples["ingredients"])
samples = pd.concat([samples.reset_index(drop=True), ingredients_df.reset_index(drop=True)], axis=1).drop("ingredients", axis=1)

def percent_to_numeric(percent_str):
    try:
        return pd.to_numeric(percent_str.replace('%', ''))
    except ValueError:
        return None

samples["cocoa_percent"] = samples["cocoa_percent"].apply(percent_to_numeric)

def split_and_strip_characteristics(df):
    df_long = df['characteristics'] \
    .str.split(',') \
    .explode() \
    .str.strip() \
    .reset_index(name='characteristic') \
    .rename(columns={'index': 'sample_id'})

    return df_long

tastes = split_and_strip_characteristics(tastes)

def order_col_by_rating(df, col):
    result_df = df \
        .groupby(col)['rating'] \
        .mean() \
        .reset_index() \
        .sort_values(by='rating', ascending=False) 
    return result_df

# best origins
ordered_origins = order_col_by_rating(samples, 'bean_origin')
ordered_origins_path = os.path.join(project_path, 'data', 'ordered_origins.tsv')
ordered_origins.to_csv(ordered_origins_path, sep='\t', index=False)

# best characteristic
sample_tastes = pd.merge(tastes, samples, on='sample_id')
ordered_tastes = order_col_by_rating(sample_tastes, 'characteristic')
ordered_tastes_path = os.path.join(project_path, 'data', 'ordered_tastes.tsv')
ordered_tastes.to_csv(ordered_tastes_path, sep='\t', index=False)
# ingredient ratings
ingredient_ratings = pd.DataFrame(columns=['ingredient', 't_statistic', 'p_value'])
ingredient_names = ['beans', 'sugar', 'sweetener', 'cocoa butter', 'vanilla', 'lecithin', 'salt']
for ingr in ingredient_names:

    with_ingr = samples[samples[ingr] == True]['rating']
    without_ingr = samples[samples[ingr] == False]['rating']

    t_statistic, p_value = ttest_ind(with_ingr, without_ingr, equal_var=False)

    threshold = 1e-2
    result_row = pd.DataFrame({
        'ingredient': [ingr],
        't_statistic': [t_statistic],
        'p_value': [p_value],
        'significant': [p_value > threshold]
    })

    ingredient_ratings = pd.concat([ingredient_ratings, result_row], axis=0, ignore_index=True)
 
ingredient_ratings_path = os.path.join(project_path, 'data', 'ingredient_ratings.tsv')
ingredient_ratings.to_csv(ingredient_ratings_path, sep='\t', index=False)