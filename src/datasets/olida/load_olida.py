import pandas as pd
import re

# List of files to load
files = {
    "genepairs": "./root/data/olida/GeneCombination.tsv",
    "genes": "./root/data/olida/Gene.tsv",
    "variantcombinations": "./root/data/olida/Combination.tsv",
    "snv": "./root/data/olida/SMALLVARIANT.tsv",
    "cnv": "./root/data/olida/CNV.tsv"
}

# Function to load data based on file extension
def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.tsv'):
        return pd.read_csv(file_path, sep='\t')

def parse_gene_pairs():
    genepairs_df = load_data(files["genepairs"])
    genes_df = load_data(files["genes"])
    
    split_genes = genepairs_df['Genes'].str.split(';', expand=True)
    if split_genes.shape[1] >= 2:
        genepairs_df['GeneName1'], genepairs_df['GeneName2'] = split_genes[0].str.strip(), split_genes[1].str.strip()
    else:
        genepairs_df['GeneName1'], genepairs_df['GeneName2'] = 'Unknown', 'Unknown'

    # Merging gene details from the genes DataFrame
    genepairs_df = genepairs_df.merge(genes_df, left_on='GeneName1', right_on='Gene Name', how='left', suffixes=('', '_1'))
    genepairs_df = genepairs_df.merge(genes_df, left_on='GeneName2', right_on='Gene Name', how='left', suffixes=('_1', '_2'))

    # Parsing and structuring variant information
    for idx in ['1', '2']:
        genepairs_df[f'Variants_{idx}_Details'] = genepairs_df[f'Variants_{idx}'].apply(parse_variants)

    # Convert DataFrame to a list of dictionaries
    list_of_dicts = genepairs_df.to_dict(orient='records')
    return list_of_dicts

def parse_variants(variant_str):
    """Parse variant string and return structured variant details for multiple possible variants.
    Skips entries without cDNA changes."""
    if not variant_str:
        return [{'Details': 'No variant info'}]

    variants_info = []
    # Splitting the string by semicolon to handle potential multiple variants in the string
    variant_list = variant_str.split(';')
    for variant in variant_list:
        variant = variant.strip()
        if ':' not in variant:
            # Skip variants that don't contain the gene:cDNA part
            continue

        try:
            # Split by the first comma to separate gene:cDNA change from protein change if present
            parts = variant.split(',', 1)
            gene_cdna = parts[0].split(':')
            gene = gene_cdna[0].strip()
            if len(gene_cdna) < 2:
                # If there's no cDNA change, skip this variant
                continue
            cdna_change = gene_cdna[1].strip()

            # Check if there's a protein change and extract it if present
            protein_change = "Unknown"
            if len(parts) > 1:
                protein_change = parts[1].strip()
                # Removing any prefix like 'p.' which is common in protein changes
                protein_change = protein_change.split(' ')[-1] if ' ' in protein_change else protein_change

            variants_info.append({
                'Gene': gene,
                'Cdna_Change': cdna_change,
                'Protein_Change': protein_change
            })
        except Exception as e:
            print(f"Error parsing variant '{variant}': {e}")
            variants_info.append({'Details': f"Error parsing variant '{variant}': {e}"})

    return variants_info

def parse_variant_combinations():
    variant_combinations_df = load_data(files["variantcombinations"])
    snv_df = load_data(files["snv"])
    cnv_df = load_data(files["cnv"])
    variants_full_df = pd.concat([snv_df, cnv_df])

    variant_combinations_details = []
    for index, row in variant_combinations_df.iterrows():
        base_info = {
            "OLIDA_ID": row["OLIDA ID"],
            "Disease": row["Diseases"],
            "Oligogenic_Effect": row["Oligogenic Effect"]
        }
        # Container for all variants of a single row
        variants_info = {}

        for idx, variant in enumerate(row["Associated Variants"].split(';')):
            parts = variant.split(':')
            if len(parts) > 3:
                gene = parts[1]
                genomic_positions = parts[2].split(',')
                cdna_change = parts[3].split(',')[0]
                protein_change = parts[3].split(',')[1] if len(parts[3].split(',')) > 1 else "None"
                zygosity = parts[4] if len(parts) > 4 else "Unknown"

                # Process and fetch variant-specific details
                variant_details = process_variant(variants_full_df, gene, cdna_change)

                # Assign variant-specific details to a unique key for each variant
                variants_info[f"Variant_{idx+1}"] = {
                    "Gene": gene,
                    "Genomic_Position_Hg19": genomic_positions[0] if len(genomic_positions) > 0 else "N.A.",
                    "Genomic_Position_Hg38": genomic_positions[1] if len(genomic_positions) > 1 else "N.A.",
                    "Cdna_Change": cdna_change,
                    "Protein_Change": protein_change,
                    "Zygosity": zygosity,
                    **variant_details
                }
            else:
                variants_info[f"Variant_{idx+1}"] = {"Details": "Incomplete variant data"}

        # Combine the base info with all variants info under the same entry
        full_entry = {**base_info, **variants_info}
        variant_combinations_details.append(full_entry)

    return variant_combinations_details

def process_variant(variants_df, gene, cdna_change):
    cdna_change_escaped = re.escape(cdna_change)
    variant = variants_df[(variants_df['Gene'] == gene) & (variants_df['Cdna Change'].str.contains(cdna_change_escaped, na=False))]
    if not variant.empty:
        variant_info = variant.iloc[0].to_dict()
        formatted_info = {k.replace(' ', '_'): v for k, v in variant_info.items()}  # Replace spaces with underscores in keys
        return formatted_info
    return {'Details': 'Variant not found'}
