from src.dataloader.data_wrapper import ClinVarDataWrapper
import json
ALL_RECORDS = False
NUM_RECORDS = 30000
SEQ_LEN = 1024

def save_clinvar_data(data, file_path="clinvar_data.json"):
    """
    Process and save ClinVar data as a JSON file.

    Args:
        data (list): List of ClinVar records in the format [[[ref, alt, variant_type], label], ...]
        file_path (str): Path to save the JSON file.
    """
    label_map = {
        "Benign": 0,
        "Likely_benign": 1,
        "Likely_pathogenic": 2,
        "Pathogenic": 3
    }
    formatted_data = []
    for record in data:
        sequences, label = record
        ref, alt, _ = sequences
        formatted_data.append({
            "sequence": f"{ref}[MASK]{alt}",
            "label": label_map.get(label, -1)  # Default to -1 for unknown labels
        })
    with open(file_path, "w") as f:
        json.dump(formatted_data, f, indent=4)
    print(f"Data saved to {file_path}")

data_loader = ClinVarDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
save_clinvar_data(data)
