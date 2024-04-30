from mavehgvs import VariantPosition

# Example string for a genomic variant
variant_string = "c.123A>G"

# Parse the variant string
variant_pos = VariantPosition(variant_string)

# Output parsed details
print(f"Position: {variant_pos.position}")
print(f"Amino Acid: {variant_pos.amino_acid}")
print(f"Is intronic: {variant_pos.is_intronic()}")

