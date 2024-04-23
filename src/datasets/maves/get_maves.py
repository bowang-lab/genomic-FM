import requests
import csv
from io import StringIO
import pandas as pd
import re
from mavehgvs.patterns.dna import (
    dna_sub_c, dna_del_c, dna_ins_c, dna_dup_c, dna_delins_c,
    dna_sub_gmo, dna_del_gmo, dna_ins_gmo, dna_dup_gmo, dna_delins_gmo,
    dna_sub_n, dna_del_n, dna_ins_n, dna_dup_n, dna_delins_n,
    dna_multi_variant, dna_equal_c, dna_equal_gmo, dna_equal_n
)

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

def apply_change(dna_sequence, match):
    if 'delins' in match.re.pattern:
        start = int(match.group('start')) - 1
        end = int(match.group('end') if 'end' in match.groupdict() and match.group('end') else start)
        new_sequence = match.group('seq')
        dna_sequence = dna_sequence[:start] + new_sequence + dna_sequence[end:]
    elif 'del' in match.re.pattern:
        if 'start' in match.groupdict() and match.group('start'):
            start = int(match.group('start')) - 1
            end = int(match.group('end')) if 'end' in match.groupdict() and match.group('end') else start
        else:
            start = int(match.group('position')) - 1
            end = start
        dna_sequence = dna_sequence[:start] + dna_sequence[end + 1:]
    elif 'ins' in match.re.pattern:
        start = int(match.group('start')) - 1
        seq = match.group('seq')
        dna_sequence = dna_sequence[:start + 1] + seq + dna_sequence[start + 1:]
    elif 'dup' in match.re.pattern:
        if 'start' in match.groupdict() and match.group('start'):
            start = int(match.group('start')) - 1
            end = int(match.group('end')) if 'end' in match.groupdict() and match.group('end') else start
        else:
            start = int(match.group('position')) - 1
            end = start
        segment = dna_sequence[start:end + 1]
        dna_sequence = dna_sequence[:end + 1] + segment + dna_sequence[end + 1:]
    elif 'sub' in match.re.pattern:
        position = int(match.group('position')) - 1
        new_base = match.group('new')
        dna_sequence = dna_sequence[:position] + new_base + dna_sequence[position + 1:]
    return dna_sequence

def get_alternate_sequence(dna_sequence, hgvs_nt):
    prefix_patterns = {
        'c.': [dna_sub_c, dna_del_c, dna_ins_c, dna_dup_c, dna_delins_c, dna_equal_c],
        'gmo.': [dna_sub_gmo, dna_del_gmo, dna_ins_gmo, dna_dup_gmo, dna_delins_gmo, dna_equal_gmo],
        'n.': [dna_sub_n, dna_del_n, dna_ins_n, dna_dup_n, dna_delins_n, dna_equal_n]
    }

    prefix = hgvs_nt.split('.')[0] + '.'
    if prefix not in prefix_patterns:
        print(f"Error: Prefix {prefix} not recognized.")
        return None

    changes = re.findall(r'\[(.*?)\]', hgvs_nt)[0].split(';') if '[' in hgvs_nt else [hgvs_nt[len(prefix):]]
    modified_sequence = dna_sequence[:]
    for change in changes:
        applied = False
        for pattern in prefix_patterns[prefix]:
            match = re.match(pattern, change.strip())
            if match:
                modified_sequence = apply_change(modified_sequence, match)
                applied = True
                break
        if not applied:
            print(f"Error: Unable to parse change: {change}")
            return None

    return modified_sequence
