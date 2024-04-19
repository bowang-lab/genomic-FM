import os
import pprint


def process_json(file_path):
    with open(file_path, 'r') as file:
        data = json.loads(file.read())[0]  # Assuming each file contains a single JSON object in a list

        # Direct extraction for simpler fields
        processed_data = {
            'external_id': data.get('external_id', ''),
            'clinicalStatus': data.get('clinicalStatus', ''),
            'sex': data.get('sex', ''),
            'date_of_birth': f"{data.get('date_of_birth', {}).get('year', '')}-{data.get('date_of_birth', {}).get('month', ''):02d}",
            'life_status': data.get('life_status', ''),
            'maternal_ethnicity': ', '.join(data.get('ethnicity', {}).get('maternal_ethnicity', [])),
            'paternal_ethnicity': ', '.join(data.get('ethnicity', {}).get('paternal_ethnicity', [])),
            'medical_history': data.get('notes', {}).get('medical_history', ''),
            'family_history': data.get('notes', {}).get('family_history', ''),
            'indication_for_referral': data.get('notes', {}).get('indication_for_referral', ''),
            'global_mode_of_inheritance': ', '.join(data.get('global_mode_of_inheritance', [])),
            # Add more fields as needed
        }

        # Initialize columns for observed features
        for feature in data.get('features', []):
            feature_label = feature.get('label', '').replace(' ', '_').lower()  # Creating a valid column name
            processed_data[feature_label] = 0  # Initialize all features as not observed

        # Mark observed features 
        for feature in data.get('features', []):
            if feature.get('observed', 'no') == 'yes':
                feature_label = feature.get('label', '').replace(' ', '_').lower()  # Use the same label as the column name
                processed_data[feature_label] = 1  # Mark as observed

        return processed_data

def create_metadata(json_dir, out_file='./root/data/sk_cardiac_data_metadata_summary.csv'):
    if os.path.exists(out_file):
        metadata = pd.read_csv(out_file)
    else:
        metadata_summary = []
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(json_dir, filename)
                metadata_summary.append(process_json(file_path))
        # Convert the aggregated metadata into a DataFrame
        metadata = pd.DataFrame(metadata_summary)
        metadata.to_csv(outfile, index=False)
    return(metadata)

json_dir = '/cluster/projects/bwanggroup/precision_medicine/data/sk_cardiac_data/Phenotips_Data'
metadata = create_metadata(json_dir)

# Print the DataFrame to check
print(metadata.head())




