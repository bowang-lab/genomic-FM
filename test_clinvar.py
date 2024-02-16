# Import necessary functions from the modules
from src.datasets.clinvar.load_clinvar import download_file, read_vcf
from src.datasets.clinvar.filter_record import filter_records

# Download the ClinVar VCF file and store the path to the downloaded file
clinvar_vcf_path = download_file()

# Read the first 100 records from the VCF file
res = read_vcf(clinvar_vcf_path, num_records=100)

# Define a function to filter out SNP variants (Single Nucleotide Polymorphisms)
def non_SNP_variant(record):
    # A SNP is characterized by having a length of 1 for both the reference and alternate bases.
    # This function returns True for non-SNP variants.
    return len(record['Reference Base']) != 1 or len(record['Alternate Base'][0]) != 1

# Apply the filter function to the list of records
filtered_records = filter_records(res, lambda record: non_SNP_variant(record))

# Import the necessary functions for sequence extraction
from src.datasets.clinvar.extract_seq import extract_sequence
import kipoiseq

# Select the first record from the filtered records
record = filtered_records[0]
print(record)
chr = record['Chromosome']
pos = record['Position']
ref = record['Reference Base']
alt = record['Alternate Base'][0]
id = record['ID']

# Set the length of the sequence to be extracted
SEQUENCE_LENGTH = 1024

# Create a Variant object using kipoiseq, which represents the genetic variant
variant = kipoiseq.Variant(f"chr{chr}", pos, ref, alt, id=f'rs{id}')

# Extract the reference and alternate sequences surrounding the variant
reference, alternate = extract_sequence(variant, sequence_length=SEQUENCE_LENGTH)

# Print the extracted reference and alternate sequences
print(f"Reference sequence: {reference}")
print(f"Alternate sequence: {alternate}")
