import pandas as pd
import json
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Convert json to csv and only retain the tags.")
parser.add_argument("base_filename", help="The base filename without extension (e.g., 'la_mountain_trails')")
args = parser.parse_args()

# Construct input and output filenames
input_file = args.base_filename + ".json"
output_file = args.base_filename + ".csv"

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