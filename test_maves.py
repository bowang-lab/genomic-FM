from src.datasets.maves.get_maves import get_all_urn_ids, get_score_set, get_scores

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



