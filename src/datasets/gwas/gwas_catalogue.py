import pandas as pd
from pyfaidx import Fasta
import os
import subprocess
import requests
import sgkit as sg
import numpy as np
import scipy.sparse as sp_sparse

def download_file(file_path='./root/data/gwas_catalog_v1.0.2-associations_e111_r2024-03-01.tsv',
                  gwas_path='alternative'):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Starting download...")

        # URL for the file
        url = f'https://www.ebi.ac.uk/gwas/api/search/downloads/{gwas_path}'

        try:
            # Download the file
            response = requests.get(url)
            with open(file_path, 'wb') as f:
                f.write(response.content)

            print(f"File downloaded and unzipped successfully: {file_path}")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    else:
        print(f"File already exists: {file_path}")

    return file_path

def extract_variant_sequence(chromosome, position, fasta_path, flank_size=100):
    """
    Extracts the sequence around a variant position from a reference genome.

    Parameters:
    - chromosome (str): The chromosome of the variant (e.g., '1').
    - position (int): The position of the variant on the chromosome.
    - fasta_path (str): Path to the reference genome in FASTA format.
    - flank_size (int): Number of base pairs to include on each side of the variant.

    Returns:
    - str: Sequence around the variant.
    """
    # Load the reference genome
    genome = Fasta(fasta_path)
    
    # Calculate start and end positions
    start = max(1, position - flank_size)
    end = position + flank_size
    
    # Extract the sequence
    sequence = genome[f'chr{chromosome}'][start:end].seq
    
    return sequence

def get_unique_risk_snps(gwas_catalog):
    """
    Extracts all unique risk SNPs from the GWAS catalog.

    Parameters:
    - gwas_catalog (pd.DataFrame): The GWAS catalog dataframe.

    Returns:
    - list: A list of all unique risk SNPs from the GWAS catalog.
    """
    unique_snps = gwas_catalog['SNPS'].unique().tolist()
    return unique_snps

def extract_snp_details(gwas_catalog, snp):
    """
    Extracts details for a given SNP from the GWAS catalog, including the variant location,
    the allele, its beta/odds ratio, maf, all the diseases/traits it's associated with.

    Parameters:
    - gwas_catalog (pd.DataFrame): The GWAS catalog dataframe.
    - snp (str): The SNP identifier.

    Returns:
    - dict: A dictionary containing the SNP details.
    """
    # Filter the GWAS catalog for the specified SNP
    snp_info = gwas_catalog[gwas_catalog['SNPS'] == snp]
    
    # Extract the relevant details with modified risk allele information
    snp_details = {
        "Location": snp_info['CHR_ID'].astype(str) + ':' + snp_info['CHR_POS'].astype(str),
        "Value": snp_info['OR or BETA'],
        "Value Type": snp_info['OR or BETA'].apply(lambda x: "Beta" if isinstance(x, float) and x < 1 else "Odds"),
        "Traits": snp_info['DISEASE/TRAIT'],
        "Allele": snp_info['STRONGEST SNP-RISK ALLELE'].apply(lambda x: x.split('-')[-1]),
        "MAF": snp_info['RISK ALLELE FREQUENCY'],
        "Subgroup": snp_info['P-VALUE (TEXT)'].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    }
    
    # Convert each column to a unique list
    for key in snp_details:
        snp_details[key] = snp_details[key].tolist()
    
    return snp_details

def get_risk_snps(gwas_catalog, trait, pvalue_text_filter=None):
    """
    Gets all the risk SNPs associated with a given trait, optionally filtering by P-VALUE (TEXT).
    Returns their risk allele and beta/odds ratio.

    Parameters:
    - gwas_catalog (pd.DataFrame): The GWAS catalog dataframe.
    - trait (str): The trait/disease of interest.
    - pvalue_text_filter (str, optional): A keyword to filter the P-VALUE (TEXT) column.

    Returns:
    - pd.DataFrame: A dataframe containing the SNPs, their risk alleles, and beta/odds ratio
                     for the specified trait, optionally filtered by P-VALUE (TEXT).
    """
    # Filter by trait
    filtered_data = gwas_catalog[gwas_catalog['DISEASE/TRAIT'].str.contains(trait, case=False, na=False)]
    
    # If a filter is provided, apply it
    if pvalue_text_filter:
        filtered_data = filtered_data[filtered_data['P-VALUE (TEXT)'].str.contains(pvalue_text_filter, case=False, na=False, regex=True)]

    # Get allele 
    filtered_data['STRONGEST SNP-RISK ALLELE'] = filtered_data['STRONGEST SNP-RISK ALLELE'].apply(lambda x: x.split('-')[-1])

    # Remove entries with missing allele
    filtered_data['STRONGEST SNP-RISK ALLELE'].replace("?", pd.NA, inplace=True)
    filtered_data.dropna(subset=['STRONGEST SNP-RISK ALLELE'], inplace=True)

    # Select and return the relevant columns
    return filtered_data[['DISEASE/TRAIT', 'SNPS', 'STRONGEST SNP-RISK ALLELE', 'RISK ALLELE FREQUENCY','OR or BETA']]

def get_trait_mappings(gwas_catalog, gwas_trait_mappings, trait):
    """
    Finds the EFO mappings for a given trait from the GWAS Catalog by cross-referencing
    with the GWAS Catalog trait mappings data.

    Parameters:
    - gwas_catalog (pd.DataFrame): The GWAS catalog dataframe.
    - gwas_trait_mappings (pd.DataFrame): The GWAS catalog trait mappings dataframe.
    - trait (str): The trait/disease of interest.

    Returns:
    - pd.DataFrame: A dataframe with the trait and its EFO mappings.
    """
    # Filter GWAS catalog for the specified trait
    filtered_gwas_catalog = gwas_catalog[gwas_catalog['DISEASE/TRAIT'].str.contains(trait, case=False, na=False)]
    
    # Merge with trait mappings on the trait name
    merged_data = pd.merge(filtered_gwas_catalog, gwas_trait_mappings, left_on='DISEASE/TRAIT', right_on='Disease trait', how='left')
    
    # Select relevant columns for output
    result = merged_data[['DISEASE/TRAIT', 'EFO term', 'EFO URI', 'Parent term', 'Parent URI']]
    
    return result.drop_duplicates()
