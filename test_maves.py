from src.datasets.maves.get_maves import get_all_urn_ids, get_score_set, get_scores, get_alternate_sequence
import pandas as pd
from tqdm import tqdm

urn_ids = get_all_urn_ids()
print(f"Number of URN: {len(urn_ids)}")
avail = 0
total = 0
limit=len(urn_ids)

for urn_id in tqdm(urn_ids[:limit], desc="Processing URN IDs"):
    score_set = get_score_set(urn_id)
    for exp in score_set:
        print(exp)
        urn_id = exp.get('urn', None)
        numVariants = exp.get('numVariants', None)
        total += int(numVariants)  
        scores = get_scores(urn_id)

        # Check if the DataFrame 'scores' is not empty
        if not scores.empty:
            for index, row in scores.iterrows():
                if pd.notna(row['hgvs_nt']) and pd.notna(row["score"]):
                    avail += int(numVariants) 
                    alt = get_alternate_sequence(exp['targetGenes'][0]['sequence'], row['hgvs_nt'])
                    print(f'Alternate sequence: {alt} HGVS_NT: {row["hgvs_nt"]} Score: {row["score"]}')

print(f"Total number of MAVEs: {avail}/{total}")
