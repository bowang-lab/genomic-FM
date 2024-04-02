from src.datasets.maves.get_maves import get_all_urn_ids, get_score_set, get_scores, get_alternate_sequence
import pandas as pd

# Example usage
urn_ids = get_all_urn_ids()
for urn_id in urn_ids[:10]:
    print(urn_id)
    score_set = get_score_set(urn_id)
    print(score_set)
    for exp in score_set:
        urn_id = exp.get('urn', None)
        scores = get_scores(urn_id)
        scores.head()

        for index, row in scores.iterrows():
            if pd.notna(row['hgvs_nt']):
                alt = get_alternate_sequence(exp['targetGenes'][0]['sequence'],row['hgvs_nt'])
                print(f'Alternate sequence: {alt} HGVS_NT: {row["hgvs_nt"]} Score: {row["score"]}')
