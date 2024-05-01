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
from tqdm import tqdm

dna_sub_c_x = r'c\.(?P<position>\d+)(?P<ref>[ACGT])>(?P<new>[ACGTX])'
dna_sub_gmo_x = r'gmo\.(?P<position>\d+)(?P<ref>[ACGT])>(?P<new>[ACGTX])'
dna_sub_n_x = r'n\.(?P<position>\d+)(?P<ref>[ACGT])>(?P<new>[ACGTX])'

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
    try:
        mutation_type = match.lastgroup  # This depends on named groups in your regex
        if mutation_type == 'sub':
            position = int(match.group(1)) - 1
            new_base = match.group(3)
            if new_base == 'X':
                print(f"Skipping unknown substitution at position {position + 1}")
                return dna_sequence  # Return unmodified sequence or handle as needed
            else:
                dna_sequence = dna_sequence[:position] + new_base + dna_sequence[position + 1:]
        elif mutation_type in ['del', 'delins', 'dup', 'ins']:
            start = int(match.group('start')) - 1
            end = int(match.group('end')) if match.group('end') else start
            if mutation_type == 'del':
                dna_sequence = dna_sequence[:start] + dna_sequence[end + 1:]
            elif mutation_type == 'delins':
                new_sequence = match.group('seq')
                dna_sequence = dna_sequence[:start] + new_sequence + dna_sequence[end + 1:]
            elif mutation_type == 'ins':
                seq = match.group('seq')
                dna_sequence = dna_sequence[:start + 1] + seq + dna_sequence[start + 1:]
            elif mutation_type == 'dup':
                segment = dna_sequence[start:end + 1]
                dna_sequence = dna_sequence[:end + 1] + segment + dna_sequence[end + 1:]
        return dna_sequence
    except Exception as e:
        print(f"Error applying change at position {match.group('start')} for mutation type {mutation_type}: {str(e)}")
        return None

def get_alternate_dna_sequence(dna_sequence, hgvs_nt):
    prefix_patterns = {
        'c.': [dna_sub_c_x, dna_sub_c, dna_del_c, dna_ins_c, dna_dup_c, dna_delins_c, dna_equal_c],
        'gmo.': [dna_sub_gmo_x, dna_sub_gmo, dna_del_gmo, dna_ins_gmo, dna_dup_gmo, dna_delins_gmo, dna_equal_gmo],
        'n.': [dna_sub_n_x, dna_sub_n, dna_del_n, dna_ins_n, dna_dup_n, dna_delins_n, dna_equal_n]
    }

    prefix = hgvs_nt.split('.')[0] + '.'
    if prefix not in prefix_patterns:
        print(f"Error: Prefix {prefix} not recognized.")
        return None

    changes = re.findall(r'\[(.*?)\]', hgvs_nt)[0].split(';') if '[' in hgvs_nt else [hgvs_nt[len(prefix):]]
    modified_sequence = dna_sequence[:]
    for change in changes:
        if '=' in change:
            continue  # Skip processing as '=' indicates no change
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

def get_maves(Seq_length=1024, limit = None, target='score'):
    urn_ids = get_all_urn_ids()
    avail = 0
    total = 0
    n_studies = 0
    data = [] 

    for study_num, urn_id in tqdm(enumerate(urn_ids)):
        score_set = get_score_set(urn_id)

        if limit and study_num >= limit:
            break

        for index, exp in enumerate(score_set):
            if not exp.get('targetGenes'):  # Check if targetGenes is empty or not present
                print(f"Warning: No target genes found for {urn_id}")
                continue  # Skip this entry if no target genes

            print(exp)
            urn_id = exp.get('urn', None)
            title = exp.get('title', None)
            description = exp.get('description', None)
            numVariants = exp.get('numVariants', None)
            sequence_type = exp.get('targetGenes', None)[0]['sequence_type']
            annotation = ': '.join([title, description])

            total += int(numVariants)  
            scores = get_scores(urn_id)
            if isinstance(scores, pd.DataFrame) and sequence_type == "dna":
                if not scores.empty:
                    if index == 0:
                        n_studies += 1
                    for index, row in scores.iterrows():
                        if pd.notna(row['hgvs_nt']) and pd.notna(row["score"]):
                            alt = get_alternate_dna_sequence(exp['targetGenes'][0]['sequence'], row['hgvs_nt'])
                            if alt:
                                if len(exp["targetGenes"][0]["sequence"]) <= Seq_length:
                                    avail += 1 
                                    ref = exp["targetGenes"][0]["sequence"]
                                    x = [ref, alt, annotation]
                                    y = row[target]
                                    data.append([x,y])
                            else:
                                print(f"Error: Could not retrieve alternate sequence: {urn_id}")
                else:
                    print(f"Error: Scores dataframe is empty: {urn_id}")
            else:
                print(f"Error: Could not retrieve scores for: {urn_id}")
    print(f"Total number of studies: {n_studies}/{len(urn_ids)}")
    print(f"Total number of MAVEs: {avail}/{total}")
    return data