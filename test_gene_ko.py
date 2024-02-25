from src.datasets.gene_ko.get_gene_knock_out import (
    create_fitness_scores_dataframe,
    create_variant_sequence_and_reference_sequence_for_gene
)
from src.sequence_extractor import GenomeSequenceExtractor
from src.datasets.cellpassport.load_cell_passport import load_cell_line_annotations

print('Creating fitness scores dataframe...')
fitness_scores = create_fitness_scores_dataframe()

print('Creating variant sequence and reference sequence for gene...')
gene = fitness_scores.iloc[0]
print(gene)
record = create_variant_sequence_and_reference_sequence_for_gene(gene,
                                                                 insert_Ns=True)
# Another way is to delete the gene completely
# record = create_variant_sequence_and_reference_sequence_for_gene(gene,
#                                                                  insert_Ns=False)

print("extract sequence")
SEQUENCE_LENGTH = 20
genome_extractor = GenomeSequenceExtractor()
# Extract sequences
reference, alternate = genome_extractor.extract_sequence_from_record(record, SEQUENCE_LENGTH)
print(f"Reference sequence: {reference}")
print(f"Alternate sequence: {alternate}")



# Get the fitness score for the gene across all cell lines
cellline_names = gene.index[1:-3]
cellline_annotations = load_cell_line_annotations()

# Extract gene name and first cell line name
gene_name = gene[0]
cell_line_name = cellline_names[0]

# Filter annotations for the specific cell line
filtered_annotations = cellline_annotations[cellline_annotations['model_id'] == cell_line_name]

# Print the filtered annotations
print(filtered_annotations.to_string())

# Print the fitness score for the gene at the specific cell line
fitness_score = gene[1]
print(f"Fitness score for gene {gene_name} at cell line {cell_line_name}: {fitness_score}")
