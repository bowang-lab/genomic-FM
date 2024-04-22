import requests
import csv
from io import StringIO
import pandas as pd
import re
from mavehgvs.patterns.dna import (
    dna_sub_c, dna_del_c, dna_delins_c, dna_dup_c, dna_ins_c
)
from mavehgvs.patterns.util import combine_patterns

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

def combine_patterns(patterns, groupname='variant'):
    """Combines various regex patterns into a single pattern with named groups."""
    grouped_patterns = '|'.join(patterns)
    return f"(?P<{groupname}>{grouped_patterns})"

def get_alternate_sequence(dna_sequence, hgvs_nt):
    """
    Apply DNA level changes to a given sequence using sequential mavehgvs regex patterns.

    Args:
        dna_sequence (str): The original DNA sequence.
        hgvs_nt (str): Changes in HGVS format, e.g., "c.[394C>A;610T>C;611T>A;612A>T]"

    Returns:
        str: The DNA sequence with the changes applied or None if error in parsing changes.
    """

    changes = hgvs_nt.strip("c.[]").strip("n.[]").split(";")
    for change in changes:
        # Process each type of variant separately using mavehgvs patterns
        for pattern in [dna_sub_c, dna_del_c, dna_ins_c, dna_dup_c, dna_delins_c]:
            match = re.match(pattern, change)
            if match:
                if 'delins' in pattern and match.group('seq'):
                    start = int(match.group('start')) - 1
                    end = int(match.group('end')) - 1 if match.group('end') else start
                    ins_seq = match.group('seq')
                    dna_sequence = dna_sequence[:start] + ins_seq + dna_sequence[end+1:]
                elif 'del' in pattern and not 'ins' in pattern:
                    start = int(match.group('start')) - 1
                    end = int(match.group('end')) - 1 if match.group('end') else start
                    dna_sequence = dna_sequence[:start] + dna_sequence[end+1:]
                elif 'ins' in pattern and not 'del' in pattern:
                    start = int(match.group('start')) - 1
                    ins_seq = match.group('seq')
                    dna_sequence = dna_sequence[:start+1] + ins_seq + dna_sequence[start+1:]
                elif 'dup' in pattern:
                    start = int(match.group('start')) - 1
                    end = int(match.group('end')) - 1 if match.group('end') else start
                    segment = dna_sequence[start:end+1]
                    dna_sequence = dna_sequence[:end+1] + segment + dna_sequence[end+1:]
                elif 'sub' in pattern:
                    position = int(match.group('position')) - 1
                    change_to = match.group('new')
                    dna_sequence = dna_sequence[:position] + change_to + dna_sequence[position+1:]
                break  # Exit the loop after successful matching and modification
        else:
            print(f"Unable to parse change: {change}")
            return None

    return dna_sequence