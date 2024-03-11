import os
import subprocess
import requests
import zipfile
import pandas as pd

def download_file(dir_path='./root/data/dida',
                  record_id='10749489'):
    
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    # Construct URL for accessing record metadata
    metadata_url = f'https://zenodo.org/api/records/{record_id}'

    # Make a request to get the record metadata
    response = requests.get(metadata_url)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code.

    # Parse response JSON
    data = response.json()

    # Iterate over all files in the record
    for file_info in data['files']:
        file_url = file_info['links']['self']
        file_name = file_info['key']
        file_path = os.path.join(dir_path, file_name)

        if os.path.exists(file_path):
            print(f"{file_name} already exists. Skipping download.")
            continue
        
        print(f"Downloading {file_name}...")
        # Stream download to handle large files
        with requests.get(file_url, stream=True) as file_response:
            file_response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)

    print("All files have been downloaded.")

def map_digenic_variants(digenic_variants, variants):
    """
    Function to parse GeneA and GeneB, find matching variants 
    """
    # Ensure the Coordinate column in variants_full is correctly formatted
    variants['Coordinate'] = variants['Chromosome'].astype(str) + ':' + \
                                  variants['Genomic position'].astype(str) + ':' + \
                                  variants['Ref allele'] + ':' + \
                                  variants['Alt allele']
    
    mapped_variants = []
    for index, row in digenic_variants.iterrows():
        # Extract coordinates from GeneA and GeneB columns
        for gene in ['GeneA', 'GeneB']:
            coordinate = row[gene]
            match = variants[variants['Coordinate'].str.contains(coordinate, na=False)]
            if not match.empty:
                mapped_variants.append({
                    '#Combination_id': row['#Combination_id'],
                    'Gene': gene,
                    'Variant_ID': match['ID'].values[0],
                    'Coordinate': match['Coordinate'].values[0],
                    'Chromosome': match['Chromosome'].values[0],
                    'Position': match['Genomic position'].values[0],
                    'Ref allele': match['Ref allele'].values[0],
                    'Alt allele': match['Alt allele'].values[0]
                })
    return pd.DataFrame(mapped_variants)

def get_digenic_variants(digenic_variants):
    """
    Extracts all unique risk digenic variants from the DIDA.

    Parameters:
    - digenic_variants (pd.DataFrame): The DIDA dataframe.

    Returns:
    - list: A list of all unique digenic variants from DIDA.
    """
    unique_variants = digenic_variants['#Combination_id'].unique().tolist()
    return unique_variants


