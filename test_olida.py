from src.datasets.olida.load_olida import get_gene_pairs, get_variant_combinations, get_olida
from src.utils import save_as_jsonl, read_jsonl

# Example usage
gene_pairs = get_gene_pairs()
print(gene_pairs[:1])  # Display the first gene pair

variant_combinations = get_variant_combinations()
print(variant_combinations[:1])  # Display the first var combo pair

oligogenic_variants = get_olida()
save_as_jsonl(oligogenic_variants,'./root/data/oligogenic_variants.jsonl')
