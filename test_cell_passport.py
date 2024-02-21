# Import necessary functions from the modules
from src.datasets.cellpassport.load_cell_passport import download_and_extract_cell_passport_file, read_vcf
from src.datasets.clinvar.extract_seq import extract_sequence
import kipoiseq

# # Download the CellPassport VCF file and store the paths to the extracted files
cell_passport_files = download_and_extract_cell_passport_file()

print(cell_passport_files[:10])
# Read the first 100 records from the VCF file
records = read_vcf(cell_passport_files[1], num_records=100)

#--------------------------
# test sequence extraction
#--------------------------

record = records[0]

chr = record['Chromosome']
pos = record['Position']
ref = record['Reference Base']
alt = record['Alternate Base'][0]
id = record['ID']

print(record)
# Set the length of the sequence to be extracted
SEQUENCE_LENGTH = 20

# Create a Variant object using kipoiseq, which represents the genetic variant
variant = kipoiseq.Variant(chr, pos, ref, alt, id=f'rs{id}')

# Extract the reference and alternate sequences surrounding the variant
reference, alternate = extract_sequence(variant, sequence_length=SEQUENCE_LENGTH)

# Print the extracted reference and alternate sequences
print(f"Reference sequence: {reference}")
print(f"Alternate sequence: {alternate}")
