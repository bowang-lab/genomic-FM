from src.datasets.pediatric_cardiac.load_pediatric_cardiac import create_metadata
from src.sequence_extractor import GenomeSequenceExtractor

json_dir = '/cluster/projects/bwanggroup/precision_medicine/data/sk_cardiac_data/Phenotips_Data'
metadata = create_metadata(json_dir)

# Print the DataFrame to check
print(metadata.head())




