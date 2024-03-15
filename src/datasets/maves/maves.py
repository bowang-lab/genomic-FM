import requests

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

def get_scores_by_urn(urn_id):
    # Assuming there's a dedicated endpoint or logic to fetch scores based on the URN
    scores_endpoint = f"https://api.mavedb.org/api/v1/experiments/{urn_id}/score-sets" 
    response = requests.get(scores_endpoint)
    
    if response.status_code == 200:
        scores = response.json()
        # Assuming the structure of scores is straightforward and direct
        return scores
    else:
        print(f"Error: {response.status_code}")
        return {}


# Example usage
urn_ids = get_all_urn_ids()
for urn_id in urn_ids[:10]:
    print(urn_id)
    scores = get_scores_by_urn(urn_id)
    print(scores)


