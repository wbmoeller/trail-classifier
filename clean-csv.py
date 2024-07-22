import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Input and output file paths
input_file = 'la-mountain-trails.csv'
output_file = 'clean-la-mountain-trails.csv'

# Load the CSV data into a DataFrame
df = pd.read_csv(input_file)


##########
### Clean up the name column
# List of alternative name columns to check in order
alt_name_columns = ['name_1', 'alt_name', 'old_name', 'abandoned:name', 'tiger:name_base', 'tiger:name_base_1']

# Fill missing 'name' values with the first non-null value from alt_name_columns
for col in alt_name_columns:
    df['name'] = df['name'].fillna(df[col])


##########
### Drop unimportant columns

# List of columns to drop (including highly sparse columns)
columns_to_drop = ['type', 'id', 'ref', 'abandoned:highway', 'operator', 'motor_vehicle', 
                   'tiger:cfcc', 'tiger:county', 'tiger:name_base', 'tiger:name_type', 'lit',
                   'smoking', 'source', 'oneway', 'ohv', 'oneway:bicycle', 'source_ref', 'atv', 
                   'motorcar', 'note', 'motorcycle', 'sac_scale_ref', 'cutline', 'informal', 
                   'man_made', 'horse_scale', 'tiger:zip_left', 'tiger:zip_right', 'name_1', 'bridge',
                   'layer', 'maxspeed', 'website', 'cutting', 'access', 'tiger:name_base_1', 'fixme', 
                   'addr:city', 'addr:housenumber', 'addr:postcode', 'addr:street', 'loc_name', 
                   'old_name', 'abandoned:name', 'tunnel', 'tiger:reviewed', 'segregated', 
                   'crossing', 'check_date', 'construction', 'incline', 'mtb:scale', 
                   'mtb:scale:imba', 'mtb:scale:uphill', 'smoothness', 'tracktype', 'width', 'disused',
                   'dogs', 'flood_prone', 'alt_name', 'yds', 'description', 'length', 'visibility', 'highway']

# Drop the specified columns
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')


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
difficulty_order = ['hiking', 'mountain_hiking', 'demanding_mountain_hiking', 'alpine_hiking', 
                    'demanding_alpine_hiking', 'difficult_alpine_hiking']

# Create an OrdinalEncoder with the specified categories
encoder = OrdinalEncoder(categories=[difficulty_order])  

# Fit and transform the 'sac_scale' column
df['sac_scale'] = encoder.fit_transform(df[['sac_scale']])


##########
### Clean up the surface column and make it an ordianl from easiest to hardest

# Fill missing values in 'surface' with 'dirt'
df['surface'] = df['surface'].fillna('dirt')

# Replace surface values as specified
df['surface'] = df['surface'].replace({
    'earth': 'dirt',
    'compacted': 'dirt',
    'unpaved': 'dirt',
    'ground': 'dirt',
    'concrete': 'paved',
    'wood': 'dirt',
    'metal': 'dirt'
})

# Define the order of difficulty categories (easy to hard)
difficulty_order_surface = ['paved', 'gravel', 'dirt', 'rock', 'sand', 'scree']

# Create an OrdinalEncoder with the specified categories
encoder = OrdinalEncoder(categories=[difficulty_order_surface])  

# Fit and transform the 'surface' column
df['surface'] = encoder.fit_transform(df[['surface']])


##########
### Clean up the trail_visibility column and make it an ordianl from easiest to hardest

# assume if trail visibility isn't specified then it's good
df['trail_visibility'] = df['trail_visibility'].fillna('good')

# Define the order of difficulty categories (easy to hard)
difficulty_order_trail_visibility = ['excellent', 'good', 'intermediate', 'poor', 'bad', 'horrible', 'no']

# Create an OrdinalEncoder with the specified categories
encoder = OrdinalEncoder(categories=[difficulty_order_trail_visibility])  

# Fit and transform the 'surface' column
df['trail_visibility'] = encoder.fit_transform(df[['trail_visibility']])


##########
### Drop rows where 'name' is NaN (empty)
### (since we don't have a name and we dropped the reference the user wouldn't be able to look it up in any reasonable way)
df.dropna(subset=['name'], inplace=True)


##########
### Write the result

# Save the cleaned DataFrame to the new CSV file
df.to_csv(output_file, index=False)

print(f"Successfully cleaned and saved data to '{output_file}'")