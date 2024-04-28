from src.datasets.olida.load_olida import parse_gene_pairs, parse_variant_combinations

# Example usage
gene_pairs = parse_gene_pairs()
print(gene_pairs[:1])  # Display the first gene pair

variant_combinations = parse_variant_combinations()
print(variant_combinations[:1])  # Display the first variant combination
