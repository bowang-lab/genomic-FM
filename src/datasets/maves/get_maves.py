import requests
import csv
from io import StringIO
import pandas as pd
import re

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

def get_alternate_sequence(dna_sequence, hgvs_nt):
    """
    Apply DNA level changes to a given sequence.

    Args:
        dna_sequence (str): The original DNA sequence.
        hgvs_nt (str): Changes in the format "c.[394C>A;610T>C;611T>A;612A>T]"

    Returns:
        str: The DNA sequence with the changes applied.
    """
    changes = hgvs_nt.strip("c.[]").split(";")

    for change in changes:
        if 'delins' in change:
            match = re.match(r"(\d+)_?(\d+)?delins([ATCG]+)", change)
            if match:
                start, end, ins_seq = match.groups()
                start = int(start) - 1
                end = int(end) if end else start + 1
                # Validation
                expected = dna_sequence[start:end]
                if not expected:
                    print(f"Reference sequence not found for change: {change}")
                    return None
                dna_sequence = dna_sequence[:start] + ins_seq + dna_sequence[end:]
            else:
                print(f"Unable to parse change: {change}")
                return None
        elif 'del' in change and 'ins' not in change:
            match = re.match(r"(\d+)_?(\d+)?del", change)
            if match:
                start, end = match.groups()
                start = int(start) - 1
                end = int(end) if end else start + 1
                # Validation
                expected = dna_sequence[start:end]
                if not expected:
                    print(f"Reference sequence not found for deletion: {change}")
                    return None
                dna_sequence = dna_sequence[:start] + dna_sequence[end:]
            else:
                print(f"Unable to parse deletion: {change}")
                return None
        else:
            match = re.match(r"(\d+)([ATCG]+)>([ATCG]+)", change)
            if match:
                position, change_from, change_to = match.groups()
                position = int(position) - 1
                if dna_sequence[position:position+len(change_from)] == change_from:
                    dna_sequence = dna_sequence[:position] + change_to + dna_sequence[position+len(change_from):]
                    # Check if the substitution introduces a stop codon
                    codon_start = position - (position % 3)
                    codon = dna_sequence[codon_start:codon_start+3]
                    if codon in ['TAA', 'TAG', 'TGA']:
                        print(f"Stop codon introduced at position {position+1} by {change}")
                else:
                    print(f"Mismatch at position {position+1}: expected {change_from}, found {dna_sequence[position:position+len(change_from)]}")
                    return None
            else:
                print(f"Change {change} does not match expected format.")
                return None

    return dna_sequence
