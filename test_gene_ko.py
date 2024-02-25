from src.datasets.gene_ko.get_gene_knock_out import (
    create_fitness_scores_dataframe,
    create_variant_sequence_and_reference_sequence_for_gene
)
from src.sequence_extractor import GenomeSequenceExtractor


print('Creating fitness scores dataframe...')
fitness_scores = create_fitness_scores_dataframe()
print(fitness_scores)

print('Creating variant sequence and reference sequence for gene...')
gene = fitness_scores.iloc[0]
record = create_variant_sequence_and_reference_sequence_for_gene(gene,
                                                                 insert_Ns=True)
# record = create_variant_sequence_and_reference_sequence_for_gene(gene,
#                                                                  insert_Ns=False)

print("extract sequence")
SEQUENCE_LENGTH = 20
genome_extractor = GenomeSequenceExtractor()
# Extract sequences
reference, alternate = genome_extractor.extract_sequence_from_record(record, SEQUENCE_LENGTH)
print(f"Reference sequence: {reference}")
print(f"Alternate sequence: {alternate}")
