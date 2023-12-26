# importing important libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import quote, urljoin


# Set display options to show a large number of characters in each cell
pd.set_option('display.max_colwidth', None)

# create data file from csv
data = pd.read_csv("/mnt/c/Users/97252/Desktop/Final Project/ICQC_25Dec2023(post-quarantine).csv")

# checking that data is ok
data

# Save the first column as a DataFrame
clone_name = data.iloc[:, 0:1]

# Display the first column DataFrame
print(clone_name)

# Define the value of the second row and first column
firstValue = value_at_first_row_first_column = clone_name.iloc[1, 0]

# Print the value and checking that's it's OK
firstValue

# For now, we've imported the csv file and extracted only the first column into our dataframe.

# Create a new column to store the URLs
clone_name['clone_name_URL'] = ''

# Checking the 'clone_name' df...
print(clone_name)


# Base URL
base_url = 'https://www.icgd.reading.ac.uk/search_name.php?clonename='

# Iterate through the rows
for index, row in clone_name.iterrows():
    # Get the value from the first column
    value = str(row.iloc[0])
    
    # Replace space, slash, [, and ] characters
    value = value.replace(" ", "+").replace("/", "%2F").replace("[", "%5B").replace("]", "%5D")
    
    # URL encode the modified value and construct the URL
    #encoded_value = quote(value)
    url = base_url + value
    
    # Populate the 'clone_name_URL' column
    clone_name.at[index, 'clone_name_URL'] = url

# Display the updated DataFrame
print(clone_name)

# For this far, we have two columns into our data frame: 'Clone Name' + 'clone_name_URL'. Let's coninue.

# Create a new column to store the  na code URLs
clone_name['na_code_URL'] = ''

# Checking our df
print(clone_name)

# Base URL
base_data_url = 'https://www.icgd.reading.ac.uk/'





###########################################################

# Iterate through the rows
for index, row in clone_name.iterrows():
    # Get the value from the first column
    value = str(row.iloc[0])
    
    # Replace space, slash, [, and ] characters
    value = value.replace(" ", "+").replace("/", "%2F").replace("[", "%5B").replace("]", "%5D")
    
    # URL encode the modified value and construct the search URL
    search_url = base_url + value
    
    # Fetch HTML content
    response = requests.get(search_url)
    html_content = response.content
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the URL in the HTML using BeautifulSoup
    data_button = soup.find('a', {'class': 'btn-yellow'})
    if data_button and 'href' in data_button.attrs:
        relative_data_url = data_button['href']
        
        # Extract the nacode from the relative data URL
        nacode = relative_data_url.split('=')[1]
        
        # Construct the full data URL
        full_data_url = urljoin(base_data_url, 'all_data.php?nacode=' + nacode)
        
        # Populate the 'na_code_URL' column
        clone_name.at[index, 'na_code_URL'] = full_data_url

# Display the updated DataFrame
print(clone_name)

# Display the updated DataFrame
print(clone_name.to_string())


print(clone_name.head())
print(clone_name[['clone_name_URL', 'na_code_URL']])
clone_name
