import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Clean and transform trail data.")
parser.add_argument("base_filename", help="The base filename without extension (e.g., 'la_mountain_trails')")
args = parser.parse_args()

# Construct input and output filenames
input_file = args.base_filename + ".csv"
output_file = args.base_filename + "-clean.csv"

# Load the CSV data into a DataFrame
df = pd.read_csv(input_file, low_memory=False)


##########
### Clean up the name column

# If there's a name:en column, copy that into the name column
# Check if 'name:en' column exists
if 'name:en' in df.columns:
    # Create a boolean mask for rows where 'name:en' is not null
    mask = pd.notnull(df['name:en'])

    # Update 'name' values where mask is True
    df.loc[mask, 'name'] = df.loc[mask, 'name:en']

    # Drop 'name:en' column
    df.drop(columns=['name:en'], inplace=True)

# List of alternative name columns to check in order
alt_name_columns = ['name_1', 'alt_name', 'old_name', 'abandoned:name', 'tiger:name_base', 'tiger:name_base_1']

# Fill missing 'name' values with the first non-null value from alt_name_columns
for col in alt_name_columns:
    df['name'] = df['name'].fillna(df[col])


##########
### Fill missing values that are easy to guess

# Fill missing values with 'no' for the specified columns
columns_to_fill_no = ['bicycle', 'horse', 'dog', 'stroller', 'wheelchair']
df[columns_to_fill_no] = df[columns_to_fill_no].fillna('no')


##########
### Ordinal encode permissability columns (horse, dog, wheelchair, etc) where 0 is easier and 1 is harder

df['bicycle'] = df['horse'].apply(lambda x: 2 if x == 'no' else 1)
df['horse'] = df['horse'].apply(lambda x: 2 if x == 'no' else 1)
df['dog'] = df['horse'].apply(lambda x: 2 if x == 'no' else 1)
df['stroller'] = df['horse'].apply(lambda x: 1 if x == 'no' else 0)
df['wheelchair'] = df['horse'].apply(lambda x: 1 if x == 'no' else 0)


##########
### Clean up the foot column (we only care about trails where foot traffic is allowed)

# Fill missing values in 'foot' with 'yes'
df['foot'] = df['foot'].fillna('yes')

# Remove rows where 'foot' is 'no'
df = df[df['foot'] != 'no']

# Drop 'foot' column
df.drop('foot', axis=1, inplace=True)


##########
### Nake sac_scale an ordianl from easiest to hardest

# Define the order of difficulty categories (easy to hard)
difficulty_order = ['hiking', 
					'mountain_hiking', 'demanding_mountain_hiking', 
					'alpine_hiking', 'demanding_alpine_hiking', 'difficult_alpine_hiking',
					'unknown']

# Create an OrdinalEncoder with the specified categories
encoder = OrdinalEncoder(categories=[difficulty_order], handle_unknown = 'use_encoded_value', unknown_value=-1)  

# Fit and transform the 'sac_scale' column
df['sac_scale'] = encoder.fit_transform(df[['sac_scale']])


##########
### Clean up the surface column and make it an ordianl from easiest to hardest

# Fill missing values in 'surface' with 'dirt'
# df['surface'] = df['surface'].fillna('dirt')

# Replace surface values as specified
df['surface'] = df['surface'].replace({
	# types of paved surface
    'concrete': 'paved',
    'asphalt': 'paved',
    'paving_stones': 'paved',
    'sett': 'paved',
    'cobblestone': 'paved',
    'unhewn_cobblestone': 'paved',
    'concrete:lanes': 'paved',
    'concrete:plates': 'paved',
    'grass_paver': 'paved',
    'paved_and_woodchipped': 'paved',

	# types of gravel
	'pebblestone': 'gravel',
	'fine_gravel': 'gravel',

	# types of dirt
    'earth': 'dirt',
    'compacted': 'dirt',
    'unpaved': 'dirt',
    'ground': 'dirt',
    'wood': 'dirt',
    'metal': 'dirt',
	'grass': 'dirt',
	'boardwalk': 'dirt',
	'dirt/sand': 'dirt',
	'natural': 'dirt',
	'mud': 'dirt',
	'woven_mat': 'dirt',
	'plastic': 'dirt',
	'shingle': 'dirt',
	'mulch': 'dirt',
	'acrylic': 'dirt',
	'bush': 'dirt',
	'log': 'dirt',
	'metal_grid': 'dirt',
	'dirt;unpaved': 'dirt',

    # types of rock
    'stone': 'rock',
	'rocky': 'rock',
	'bare_rock': 'rock',
	'rock;dirt': 'rock',

    # types of sand

    # types of scree
    'morraine': 'scree',
    'rocks': 'scree',

    # types of glacier
    'ice': 'glacier'
})

# Define the order of difficulty categories (easy to hard)
difficulty_order_surface = ['paved', 'gravel', 'dirt', 'rock', 'sand', 'scree', 'glacier', 'unknown']

# Create an OrdinalEncoder with the specified categories
encoder = OrdinalEncoder(categories=[difficulty_order_surface], handle_unknown = 'use_encoded_value', unknown_value=-1)  

# Fit and transform the 'surface' column
df['surface'] = encoder.fit_transform(df[['surface']])


##########
### Clean up the trail_visibility column and make it an ordianl from easiest to hardest

# assume if trail visibility isn't specified then it's good
df['trail_visibility'] = df['trail_visibility'].fillna('good')

# Define the order of difficulty categories (easy to hard)
difficulty_order_trail_visibility = ['excellent', 'good', 'intermediate', 'poor', 'bad', 'horrible', 'no', 'unknown']

# Create an OrdinalEncoder with the specified categories
encoder = OrdinalEncoder(categories=[difficulty_order_trail_visibility], handle_unknown = 'use_encoded_value', unknown_value=-1)  

# Fit and transform the 'surface' column
df['trail_visibility'] = encoder.fit_transform(df[['trail_visibility']])


##########
### Drop rows where 'name' is NaN (empty)
### (since we don't have a name and we dropped the reference the user wouldn't be able to look it up in any reasonable way)
df.dropna(subset=['name'], inplace=True)


##########
### Drop all columns that aren't interesting for our analysis
columns_to_keep = ['name', 'sac_scale', 'surface', 'trail_visibility', 'bicycle', 'dog', 'horse', 'stroller', 'wheelchair']
df = df[df.columns.intersection(columns_to_keep)]


##########
### Write the result

# Save the cleaned DataFrame to the new CSV file
df.to_csv(output_file, index=False)

print(f"Successfully cleaned and saved data to '{output_file}'")