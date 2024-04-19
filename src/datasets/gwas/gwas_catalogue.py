import pandas as pd
import os
import requests
import numpy as np
from scipy.stats import norm
import re

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

    data = pd.read_csv(file_path, sep='\t', low_memory=False)
    return data

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

def extract_snp_details(gwas_catalog, rssnp, trait=None):
    """
    Extracts details for a given SNP from the GWAS catalog, including the variant location,
    the allele, its beta/odds ratio, MAF, and all the diseases/traits it's associated with.
    An optional trait parameter allows filtering for a specific trait.

    Parameters:
    - gwas_catalog (pd.DataFrame): The GWAS catalog dataframe.
    - rssnp (str): The rsSNP identifier.
    - trait (str, optional): The specific trait to filter for.

    Returns:
    - dict: A dictionary containing the SNP details.
    """
    # Filter the GWAS catalog for the specified SNP
    snp_info = gwas_catalog[gwas_catalog['SNPS'] == rssnp]
    
    # If a specific trait is provided, further filter the DataFrame
    if trait is not None:
        snp_info = snp_info[snp_info['DISEASE/TRAIT'].str.contains(trait, case=False, na=False)]
    
    ref, alt = get_alleles_ensembl(rssnp)

    # Ensure there are SNP details to return
    if ref and not snp_info.empty:
        # Extract the relevant details with modified risk allele information
        snp_details = {
            "rsSNP": rssnp,
            "Chromosome": snp_info['CHR_ID'].astype(str).unique().tolist()[0],
            "Position": snp_info['CHR_POS'].astype(str).unique().tolist()[0],
            "Reference": ref,
            "Alternate": alt,
            "Initial Sample Size": snp_info['INITIAL SAMPLE SIZE'].tolist(),
            "Replication Sample Size": snp_info['REPLICATION SAMPLE SIZE'].tolist(),
            "Value": snp_info['OR or BETA'].tolist(),
            "Value Type": snp_info['OR or BETA'].apply(lambda x: "Beta" if isinstance(x, float) and x < 1 else "Odds").tolist(),
            "Traits": snp_info['DISEASE/TRAIT'].tolist(),
            "Risk Allele": snp_info['STRONGEST SNP-RISK ALLELE'].apply(lambda x: x.split('-')[-1]).tolist(),
            "MAF": snp_info['RISK ALLELE FREQUENCY'].tolist(),
            "P-Value": snp_info['P-VALUE'].tolist(),
            "95% CI": snp_info['95% CI (TEXT)'].tolist()
        }
        return snp_details
    else:
        print("No details found for rsSNP: {rssnp} with trait filter: {trait}")
        return None

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
    filtered_data.dropna(subset=['CHR_POS','DISEASE/TRAIT', 'SNPS', 'STRONGEST SNP-RISK ALLELE', 'RISK ALLELE FREQUENCY','95% CI (TEXT)', 'OR or BETA'], inplace=True)

    # Select and return the relevant columns
    return filtered_data[['DISEASE/TRAIT', 'SNPS', 'STRONGEST SNP-RISK ALLELE', 'RISK ALLELE FREQUENCY','95% CI (TEXT)', 'OR or BETA','P-VALUE']]

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


def parse_ci(ci_text):
    """Parse confidence intervals from a string format like '[1.04-1.16]' and return the standard error.
    
    Assumes a 95% confidence interval under a normal distribution.
    
    Parameters:
        ci_text (str): The confidence interval in string format, e.g., '[1.04-1.16]'.
        
    Returns:
        float: The estimated standard error or None if parsing fails.
    """
    try:
        # Remove the square brackets and split by '-' to get lower and upper CI bounds
        match = re.search(r'\[(.*?)\]', ci_text)
        if not match:
            raise ValueError("No text found in square brackets")

        # Split the extracted string by '-' to get lower and upper CI bounds
        lower, upper = map(float, match.group(1).split('-'))

        midpoint = (lower + upper) / 2
        half_width = (upper - lower) / 2  # The half-width of the confidence interval
        
        # Assuming a normal distribution and 95% CI, thus 1.96 * SE is approximately equal to half-width
        standard_error = half_width / 1.96
        return standard_error
    except Exception as e:
        print(f"Error parsing CI: {str(e)}")
        return None

def get_summary_stats_for_snp(snp_details, trait, value_type='Beta'):
    """ Calculates the summary statistics for a given SNP across multiple studies for specified value type (Beta or Odds). """
    # Finding indices with trait
    indices = [i for i, search_trait in enumerate(snp_details['Traits']) if trait.lower() in trait.lower()]       
    values = []
    ses = []  # List to store standard errors

    for value, v_type, ci_text in zip([snp_details['Value'][i] for i in indices], [snp_details['Value Type'][i] for i in indices], [snp_details['95% CI'][i] for i in indices]):
        if v_type == value_type and isinstance(value, (int, float)):  # Checking for correct value type and valid data
            values.append(value)
            se = parse_ci(ci_text)
            if se:
                ses.append(se)
    if not values or not ses:  # Handle cases where no valid data is found
        return None

    weights = 1 / np.square(ses)
    weighted_mean_value = np.sum(weights * values) / np.sum(weights)
    variance_weighted_mean = 1 / np.sum(weights)
    se_weighted_mean = np.sqrt(variance_weighted_mean)
    
    z_score = weighted_mean_value / se_weighted_mean
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))  # two-tailed p-value

    return {
        'rsSNP': snp_details['rsSNP'],
        'trait': trait,
        'Value Type': value_type,
        'Weighted Mean Value': weighted_mean_value,
        'SE Weighted Mean': se_weighted_mean,
        'Z-Score': z_score,
        'P-Value': p_value
    }

def get_alleles_ensembl(rs_id):
    server = "https://rest.ensembl.org"
    ext = f"/variation/human/{rs_id}?content-type=application/json"
    response = requests.get(server+ext)
    if response.status_code == 200:
        data = response.json()
        mappings = data.get('mappings', [])
        
        # Initialize variables to hold ref and alt alleles
        ref_allele = None
        alt_alleles = []

        # Process each mapping to extract ref and alt alleles
        for mapping in mappings:
            allele_string = mapping.get('allele_string')
            if allele_string:
                alleles = allele_string.split('/')
                if len(alleles) == 2:
                    # Assuming the first allele is the reference and the second is the alternate
                    ref_allele = alleles[0]
                    alt_alleles.append(alleles[1])

        # Remove duplicate alleles in case there are multiple mappings with the same allele_string
        alt_alleles = list(set(alt_alleles))
        
        return ref_allele, alt_alleles
    else:
        print(f"Failed to retrieve data for {rs_id}")
        return None, []


