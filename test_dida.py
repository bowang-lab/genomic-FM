# Import necessary functions from the modules
from src.datasets.dida.load_dida import download_file, get_digenic_variants
from src.sequence_extractor import GenomeSequenceExtractor
from pyliftover import LiftOver
import pandas as pd

# Load the DIDA data
download_file(dir_path='./root/data/dida',
                  record_id='10749489')

digenic_variants = get_digenic_variants()

SEQUENCE_LENGTH = 20
genome_extractor = GenomeSequenceExtractor()

for id in digenic_variants['#Combination_id'].unique().tolist():
    variant = digenic_variants[digenic_variants['#Combination_id'].str.contains(id, na=False)]

    if len(variant.index) == 2:
        print(f"\n[Digenic variant: {id}]")
        for index, row in variant.iterrows():
            # Perform the conversion
            chromosome="chr"+row['Chromosome']
            position=row['Position']

            lo = LiftOver('hg19', 'hg38')
            
            converted_coords = lo.convert_coordinate(chromosome,position)

            # Check if conversion was successful and print the result
            if converted_coords:
                new_chrom, new_pos, _, _ = converted_coords[0]

                record = {
                    'Chromosome': new_chrom, 
                    'Position': new_pos,
                    'Reference Base': row['Ref allele'],  
                    'Alternate Base': row['Alt allele'],  
                    'ID': row['#Combination_id']
                }
                reference, alternate = genome_extractor.extract_sequence_from_record(record, SEQUENCE_LENGTH)
                print(f"Reference sequence for {row['Gene']} : {reference}")
                print(f"Alternate sequence for {row['Gene']} : {alternate}")

            else:
                print("Conversion failed for the given coordinate.")

