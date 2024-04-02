import requests
import csv
from io import StringIO
import pandas as pd

def get_all_urn_ids():
    experiments_endpoint = "https://api.mavedb.org/api/v1/experiments/"
    response = requests.get(experiments_endpoint)
    
    if response.status_code == 200:
        data = response.json()
        # Adjusting to match the JSON structure you provided
        urn_ids = [item['urn'] for item in data]
        urn_ids = [item for item in urn_ids if item.startswith("urn:")]
        return urn_ids
    else:
        print(f"Error: {response.status_code}")
        return []

def extract_target_info(json_data):
    results = []
    for item in json_data:
        target_info = {
            'title': item.get('title', ''),
            'description': item.get('shortDescription', ''),
            'urn': item.get('urn', ''),
            'numVariants': item.get('numVariants', ''),
            'targetGenes': []
        }

        for gene in item.get('targetGenes', []):
            # Initialize empty identifiers
            ensembl_id = None
            uniprot_id = None

            # Iterate through external identifiers and extract Ensembl and UniProt IDs
            for identifier in gene.get('externalIdentifiers', []):
                if identifier.get('identifier', {}).get('dbName') == 'Ensembl':
                    ensembl_id = identifier.get('identifier', {}).get('identifier')
                elif identifier.get('identifier', {}).get('dbName') == 'UniProt':
                    uniprot_id = identifier.get('identifier', {}).get('identifier')

            target_gene_info = {
                'gene_name': gene.get('name', ''),
                'sequence': gene.get('targetSequence', {}).get('sequence', ''),
                'sequence_type': gene.get('targetSequence', {}).get('sequenceType', ''),
                'reference_genome': gene.get('targetSequence', {}).get('reference', {}).get('shortName', ''),
                'Ensembl': ensembl_id,
                'UniProt': uniprot_id
            }
            target_info['targetGenes'].append(target_gene_info)

        results.append(target_info)

    return results

def get_scores(urn_id):
    scores_endpoint = f"https://api.mavedb.org/api/v1/score-sets/{urn_id}/scores"
    response = requests.get(scores_endpoint)
    if response.status_code == 200:
        csv_text = response.text
        csv_file = StringIO(csv_text)
        scores = pd.read_csv(csv_file)
        return scores
    else:
        print(f"Error: {response.status_code}")
        return {}

def get_score_set(urn_id):
    # Assuming there's a dedicated endpoint or logic to fetch scores based on the URN
    scores_endpoint = f"https://api.mavedb.org/api/v1/experiments/{urn_id}/score-sets" 
    response = requests.get(scores_endpoint)
    if response.status_code == 200:
        scores = response.json()
        # Assuming the structure of scores is straightforward and direct
        return extract_target_info(scores)
    else:
        print(f"Error: {response.status_code}")
        return {}

def extract_variant_details(scores):
    variant_details = []
    
    # Iterate over each row in the DataFrame
    for index, row in scores.iterrows():
        # Extract details for each variant
        variant_info = {
            'accession': row['accession'],
            'hgvs_nt': row['hgvs_nt'],
            'hgvs_pro': row['hgvs_pro'],
            'score': row['score'],
            'sd': row['sd'],
            'exp.score': row['exp.score'],
            'exp.sd': row['exp.sd'],
        }
        
        # Append the extracted information to the variant_details list
        variant_details.append(variant_info)
    
    return variant_details


# Example usage
urn_ids = get_all_urn_ids()
for urn_id in urn_ids[:10]:
    print(urn_id)
    score_set = get_score_set(urn_id)
    print(score_set)
    for exp in score_set:
        urn_id = exp.get('urn', None)
        scores = get_scores(urn_id) 
        print(scores)


