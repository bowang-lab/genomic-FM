# Import necessary functions from the modules
from src.datasets.gwas.gwas_catalogue import download_file, get_trait_mappings, get_unique_risk_snps, extract_snp_details, get_risk_snps
from src.sequence_extractor import GenomeSequenceExtractor
import pandas as pd

# Load the GWAS catalog data

gwas_file_path='./root/data/gwas_catalog_v1.0.2-associations_e111_r2024-03-01.tsv'
trait_file_path='./root/data/gwas_catalog_trait-mappings_r2024-03-01.tsv'

download_file(file_path=gwas_file_path,
                  gwas_path='alternative')
download_file(file_path=trait_file_path,
                  gwas_path='trait_mappings')

gwas_catalog = pd.read_csv(gwas_file_path, sep='\t', low_memory=False)
gwas_trait_mappings = pd.read_csv(trait_file_path, sep='\t', low_memory=False)

trait = "height"
trait_mappings = get_trait_mappings(gwas_catalog, gwas_trait_mappings, trait)
print(trait_mappings[:10])

# Set the length of the sequence to be extracted
SEQUENCE_LENGTH = 20
genome_extractor = GenomeSequenceExtractor()

unique_risk_snps = get_unique_risk_snps(gwas_catalog)

# Display the number of unique SNPs and the first few SNPs as a sample
print(f"Total unique risk SNPs found: {len(unique_risk_snps)}")

# Get rsSNPs associated with a trait
risk_snps = get_risk_snps(gwas_catalog, trait)
print(risk_snps)
    
for index, row in risk_snps.iterrows():
    rsSNP = row['SNPS']
    
    print(rsSNP)

    # Get information about a rssnp 
    snp_details = extract_snp_details(gwas_catalog, rsSNP, trait)
    print(snp_details)

    record = {
        'Chromosome': snp_details['Chromosome'], 
        'Position': int(snp_details['Position']),
        'Reference Base': snp_details['Reference'],  
        'Alternate Base': [snp_details['Risk Allele'][0]],  # Adjust as needed
        'ID': rsSNP
    }

    reference, alternate = genome_extractor.extract_sequence_from_record(record, SEQUENCE_LENGTH)
    print(f"Reference sequence for {rsSNP}: {reference}")
    print(f"Alternate sequence for {rsSNP}: {alternate}")


