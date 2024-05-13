mport pandas as pd
import re
from pyliftover import LiftOver
from src.sequence_extractor import GenomeSequenceExtractor
import random
from tqdm import tqdm

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

def get_gene_pairs():
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

def get_variant_combinations():
    variant_combinations_df = load_data(files["variantcombinations"])
    snv_df = load_data(files["snv"])
    cnv_df = load_data(files["cnv"])
    variants_full_df = pd.concat([snv_df, cnv_df])

    variant_combinations_details = []
    for index, row in variant_combinations_df.iterrows():
        base_info = {
            "OLIDA_ID": row["OLIDA ID"],
            "Disease": row["Diseases"],
            "Oligogenic_Effect": row["Oligogenic Effect"],
            "FINALmeta": int(row["FINALmeta"]),
            "FUNmeta": int(row["FUNmeta"]),
            "VARmeta": int(row["VARmeta"]),
            "GENEmeta": int(row["GENEmeta"]),
            "STATmeta": int(row["STATmeta"])
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

def extract_negative_pairs(extractor, num_pairs, length_range=(200, 1000), min_separation=1000000):
    negative_pairs = []
    while len(negative_pairs) < num_pairs:
        # Extract two sequences
        sequences = extractor.extract_random_sequence(length_range=length_range, num_sequences=2)
        if len(sequences) < 2:
            continue  # In case the extraction fails or does not return enough sequences
        seq1, seq2 = sequences[0], sequences[1]
        # Check if they are negative pairs:
        # They should either be on different chromosomes or on the same chromosome but far apart
        if seq1[0].chrom != seq2[0].chrom or abs(seq1[0].start - seq2[0].start) >= min_separation:
            negative_pairs.append((seq1, seq2))
    return negative_pairs

def load_and_process_negative_pairs(file_path='./root/data/1KGP_negative_pairs.txt', Seq_length=20, num_records=None):
    # Load the data
    data = []
    data_1kgp = pd.read_csv(file_path, delimiter='\t', index_col=False)

    # Initialize the LiftOver object for hg19 to hg38
    lo = LiftOver('hg19', 'hg38')
    genome_extractor = GenomeSequenceExtractor()

    # Iterate through each row in the DataFrame
    for _, row in data_1kgp.iterrows():
        try:
            variant_combo_alternate = [] 
            variant_combo_reference = []
            for gene_variant in ['GeneA_variant', 'GeneB_variant']:
                # Check if the gene_variant data is not NaN and is a string
                if pd.isna(row[gene_variant]) or not isinstance(row[gene_variant], str):
                    raise ValueError(f"Invalid or missing variant data in row: {row}")
                
                # Extracting chromosome and position data
                chrom, pos, ref, alt, zygosity = row[gene_variant].split(':')
                
                pos = int(pos) - 1  # Convert to 0-based for pyliftover

                # Convert hg19 to hg38
                converted = lo.convert_coordinate('chr' + chrom, pos)
                if not converted:
                    raise ValueError(f"Conversion failed for {chrom}:{pos+1}")

                # Get the new chromosome and position
                new_chrom, new_pos, _, _ = converted[0]
                new_pos += 1  # Convert back to 1-based
                
                # Create record for sequence extraction
                record = {
                    'Chromosome': new_chrom.replace('chr', ''),  # Remove 'chr' if not needed
                    'Position': new_pos,
                    'Reference Base': ref,
                    'Alternate Base': alt,
                    'ID': row['Combination_ID']
                }
                # Extract sequences
                reference, alternate = genome_extractor.extract_sequence_from_record(record, Seq_length)

                variant_combo_reference.append(reference)
                variant_combo_alternate.append(alternate)

            variant_combo_reference = 'N'.join(variant_combo_reference)
            variant_combo_alternate = 'N'.join(variant_combo_alternate)

            # Append processed data (assuming negative examples have a label '0')
            x = [variant_combo_reference, variant_combo_alternate, "1000 Genome Project"]
            y = 0
            data.append([x, y])

        except Exception as e:
            print(f"Skipping due to error: {e}")
            continue

    if num_records:
        return data[:num_records]

    return data

def get_olida(Seq_length, limit=None):
    variant_combinations = get_variant_combinations()
    data = []
    for index, variant_combo in tqdm(enumerate(variant_combinations)):
        if limit and index >= limit:
            break
        
        variant_combo_reference = []
        variant_combo_alternate = []
        if variant_combo['FINALmeta'] >= 1:
            try:
                for variant in ['Variant_1', 'Variant_2']:
                    # Check if key elements are present in the variant data
                    if all(key in variant_combo[variant] for key in ['Chromosome', 'Genomic_Position_Hg38', 'Ref_Allele', 'Alt_Allele']) and \
                        all(variant_combo[variant][key] != "N.A." and not (isinstance(variant_combo[variant][key], float)) for key in ['Chromosome', 'Genomic_Position_Hg38', 'Ref_Allele', 'Alt_Allele']):
                        record = {
                                'Chromosome': variant_combo[variant]['Chromosome'],
                                'Position': int(variant_combo[variant]['Genomic_Position_Hg38']),
                                'Reference Base': variant_combo[variant]['Ref_Allele'],
                                'Alternate Base': variant_combo[variant]['Alt_Allele'],
                                'ID': variant_combo['OLIDA_ID']
                            }
                        
                        genome_extractor = GenomeSequenceExtractor(Seq_length)
                        reference, alternate = genome_extractor.extract_sequence_from_record(record, Seq_length)
                        variant_combo_reference.append(reference)
                        variant_combo_alternate.append(alternate)
                    else:
                        raise ValueError("Missing required variant information")

                variant_combo_reference = 'N'.join(variant_combo_reference)
                variant_combo_alternate = 'N'.join(variant_combo_alternate)
                x, y = [variant_combo_reference, variant_combo_alternate, variant_combo['Disease']], 1
                data.append([x, y])

            except Exception as e:
                print(f"Skipping variant combination due to error: {e}")
                continue  

    negative_examples = load_and_process_negative_pairs(Seq_length=Seq_length)
    data += negative_examples
    random.shuffle(data)
    return data
