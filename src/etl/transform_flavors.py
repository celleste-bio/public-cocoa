'''
transforming cocoa flavors data for database
'''

# packages
import os
import pandas as pd


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

def percent_to_numeric(percent_str):
    try:
        return pd.to_numeric(percent_str.replace('%', ''))
    except ValueError:
        return None
    

def split_and_strip_characteristics(df):
    df_long = df['characteristics'] \
    .str.split(',') \
    .explode() \
    .str.strip() \
    .reset_index(name='characteristic') \
    .rename(columns={'index': 'sample_id'})

    return df_long


def order_col_by_rating(df, col):
    result_df = df \
        .groupby(col)['rating'] \
        .mean() \
        .reset_index() \
        .sort_values(by='rating', ascending=False) 
    return result_df
    
def transform_flavors(flavors_file):
    flavors = pd.read_csv(flavors_file, delimiter='\t')
    flavors["sample_id"] = flavors.index

    # build references table
    references = flavors[['REF', 'Review Date']] \
        .rename(columns={'REF': 'ref_code', 'Review Date': 'year'}) \
        .drop_duplicates()
    
    # build companies table
    companies = flavors[['Company (Manufacturer)', 'Company Location']] \
        .rename(columns={'Company (Manufacturer)': 'name', 'Company Location': 'location'}) \
        .drop_duplicates()
    
    # build tastes table
    tastes = flavors[['sample_id', 'Most Memorable Characteristics']].drop_duplicates() \
        .rename(columns={'Most Memorable Characteristics': 'characteristics'}) \
        .drop_duplicates()
    
    tastes = split_and_strip_characteristics(tastes)

    # build samples table
    samples = flavors[['sample_id', 'REF', 'Company (Manufacturer)', 'Country of Bean Origin', 'Specific Bean Origin or Bar Name', 'Ingredients', 'Cocoa Percent', 'Rating']] \
        .rename(columns={'Country of Bean Origin': 'bean_origin', 'REF': 'ref_code', 'Company (Manufacturer)': 'company', 'Specific Bean Origin or Bar Name': 'bar_name', 'Ingredients': 'ingredients', 'Cocoa Percent': 'cocoa_percent', 'Rating': 'rating'}) \
        .drop_duplicates()
    
    most_common_ingredients = samples["ingredients"].value_counts().reset_index().loc[0, "ingredients"]
    samples.replace({' ': most_common_ingredients}, inplace=True)

    ingredients_df = break_ingredients(samples["ingredients"])
    samples = pd.concat([samples.reset_index(drop=True), ingredients_df.reset_index(drop=True)], axis=1).drop("ingredients", axis=1)

    samples["cocoa_percent"] = samples["cocoa_percent"].apply(percent_to_numeric)\
    
    return samples, tastes, companies, references

# run as a script
if __name__ == "__main__":
    script_path = os.path.realpath(__file__)
    project_path = os.path.dirname(os.path.dirname(script_path))
    flavors_file = os.path.join(project_path, "cocoa_ratings.tsv")
    samples, tastes, companies, references = transform_flavors(flavors_file)
