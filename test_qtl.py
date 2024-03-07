from src.sequence_extractor import GenomeSequenceExtractor
from src.datasets.qtl.qtl_loader import download_file, extract_tar, remove_file, download, process_eqtl_data, process_sqtl_data

sequence_length = 20
genome_extractor = GenomeSequenceExtractor()

#------------------------------------------------
# eqtl
#------------------------------------------------
records = process_eqtl_data(organism="Adipose_Subcutaneous")
row = records.iloc[0]
record = row['record']
slop = row['slope']
p_val = row['pval_nominal']

reference, alternate = genome_extractor.extract_sequence_from_record(record, sequence_length)
print(f"Reference: {reference}, Alternate: {alternate}")

#------------------------------------------------
# sqtl
#------------------------------------------------
records = process_sqtl_data(organism="Adipose_Subcutaneous")
row = records.iloc[0]
record = row['record']
splice_position = row['phenotype_id']
slop = row['slope']
p_val = row['pval_nominal']

reference, alternate = genome_extractor.extract_sequence_from_record(record, sequence_length)
print(f"Reference: {reference}, Alternate: {alternate}")
