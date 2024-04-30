from src.datasets.maves.load_maves import get_all_urn_ids, get_score_set, get_scores, get_alternate_dna_sequence
import pandas as pd
from tqdm import tqdm
from src.utils import save_as_jsonl, read_jsonl

Seq_length=1024
urn_ids = get_all_urn_ids()
avail = 0
total = 0
n_studies = 0
limit=500
data = [] 
print(f"Number of URN: {limit}")

for urn_id in tqdm(urn_ids[:limit], desc="Processing URN IDs"):
    score_set = get_score_set(urn_id)

    for exp in score_set:
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
                n_studies += 1
                for index, row in scores.iterrows():
                    if pd.notna(row['hgvs_nt']) and pd.notna(row["score"]):
                        alt = get_alternate_dna_sequence(exp['targetGenes'][0]['sequence'], row['hgvs_nt'])
                        if alt:
                            if len(exp["targetGenes"][0]["sequence"]) <= Seq_length:
                                avail += 1 
                                ref = exp["targetGenes"][0]["sequence"]
                                x = [ref, alt, annotation]
                                y = row['score']
                                data.append([x,y])
                                print(f'Annotation: {annotation}')
                                print(f'HGVS_NT: {row["hgvs_nt"]} Score: {row["score"]}')
                                print(f'Reference sequence: {ref} \nAlternate sequence: {alt}')
                        else:
                            print(f"Error: Could not retrieve alternate sequence: {urn_id}")
                    else:
                        print(f"Error: Missing values: {urn_id}")
            else:
                print(f"Error: Scores dataframe is empty: {urn_id}")
        else:
            print(f"Error: Could not retrieve scores for: {urn_id}")

print(f"Total number of studies: {n_studies}/{limit}")
print(f"Total number of MAVEs: {avail}/{total}")


save_as_jsonl(data,'./root/data/maves.jsonl')
