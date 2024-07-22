import pandas as pd
import json

# Input and output filenames
input_file = 'la-mountain-trails.json'
output_file = 'la-mountain-trails.csv'

# Load JSON data
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract elements and filter for those with tags
elements_with_tags = [elem for elem in data['elements'] if 'tags' in elem]

# Normalize the filtered data, specifying `max_level=0`
df = pd.json_normalize(elements_with_tags, max_level=0) 

# Focus on the tags column (which is now a dictionary in each row)
tags_df = df['tags'].apply(pd.Series)

# Combine the original df and the expanded tags DataFrame
df = pd.concat([df.drop(['tags'], axis=1), tags_df], axis=1)

# Save as CSV with the specified filename
df.to_csv(output_file, index=False)

print(f"Successfully converted '{input_file}' to '{output_file}'!")