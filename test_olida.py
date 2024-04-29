from src.datasets.olida.load_olida import get_gene_pairs, get_variant_combinations, load_and_process_negative_pairs

# Example usage
gene_pairs = get_gene_pairs()
print(gene_pairs[:1])  # Display the first gene pair

variant_combinations = get_variant_combinations()
print(variant_combinations[:1])  # Display the first variant combination

negative_combinations = load_and_process_negative_pairs()
print(negative_combinations[:1]) 