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

# Define a function to parse GeneA and GeneB, find matching variants in variants_full, and collect the results
def map_digenic_variants(digenic_variants, variants_full):
    
    # Ensure the Coordinate column in variants_full is correctly formatted
    variants_full['Coordinate'] = variants_full['Chromosome'].astype(str) + ':' + \
                                  variants_full['Genomic position'].astype(str) + ':' + \
                                  variants_full['Ref allele'] + ':' + \
                                  variants_full['Alt allele']
    
    # Split 'GeneA' and 'GeneB' into separate rows while keeping their original index
    genes_expanded = digenic_variants.set_index('#Combination_id')[['GeneA', 'GeneB']].stack().reset_index()
    genes_expanded.columns = ['#Combination_id', 'Gene', 'Coordinate']
    
    # Merge expanded digenic variants with the full variants on 'Coordinate'
    merged_data = pd.merge(genes_expanded, variants_full, on='Coordinate', how='inner')
    
    # Rename and select relevant columns
    mapped_variants = merged_data.rename(columns={
        'ID': 'Variant_ID',
        'Gene': 'Gene',
        'Chromosome': 'Chromosome',
        'Genomic position': 'Position',
        'Ref allele': 'Ref allele',
        'Alt allele': 'Alt allele'
    }).loc[:, ['#Combination_id', 'Gene', 'Variant_ID', 'Coordinate', 'Chromosome', 'Position', 'Ref allele', 'Alt allele']]

    return mapped_variants


