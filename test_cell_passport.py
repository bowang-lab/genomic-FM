# Import necessary functions from the modules
from src.datasets.cellpassport.load_cell_passport import download_and_extract_cell_passport_file, read_vcf
from src.datasets.clinvar.extract_seq import extract_sequence
from src.sequence_extractor import GenomeSequenceExtractor

# # Download the CellPassport VCF file and store the paths to the extracted files
cell_passport_files = download_and_extract_cell_passport_file()

print(cell_passport_files[:10])
# Read the first 100 records from the VCF file
records = read_vcf(cell_passport_files[1], num_records=100)

#--------------------------
# test sequence extraction
#--------------------------

record = records[0]
print(record)
# Set the length of the sequence to be extracted
SEQUENCE_LENGTH = 20
genome_extractor = GenomeSequenceExtractor()
# Extract sequences
reference, alternate = genome_extractor.extract_sequence_from_record(record, SEQUENCE_LENGTH)


# Print the extracted reference and alternate sequences
print(f"Reference sequence: {reference}")
print(f"Alternate sequence: {alternate}")
