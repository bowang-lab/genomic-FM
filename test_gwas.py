# Import necessary functions from the modules
from src.datasets.gwas.load_gwas_catalogue import download_file, get_unique_risk_snps, extract_snp_details, get_risk_snps, get_summary_stats_for_snp 
from src.sequence_extractor import GenomeSequenceExtractor
import pandas as pd

# Load the GWAS catalog data

gwas_file_path='./root/data/gwas_catalog_v1.0.2-associations_e111_r2024-03-01.tsv'
trait_file_path='./root/data/gwas_catalog_trait-mappings_r2024-03-01.tsv'

gwas_catalog = download_file(file_path=gwas_file_path,
                  gwas_path='alternative')
gwas_trait_mappings = download_file(file_path=trait_file_path,
                  gwas_path='trait_mappings')

# All traits
traits = gwas_trait_mappings['Disease trait'].nunique()
print(f"Number of traits: {traits}")
disease_to_efo = gwas_trait_mappings.set_index('Disease trait')['EFO term'].to_dict()

# Set the length of the sequence to be extracted
SEQUENCE_LENGTH = 20
genome_extractor = GenomeSequenceExtractor()

# Number of unique rsSNPs
print(f"Total unique risk SNPs found: {len(gwas_catalog['SNPS'])}")

unique_risk_snps = get_unique_risk_snps(gwas_catalog)

# Display the number of unique SNPs and the first few SNPs as a sample
print(f"Total unique risk SNPs found: {len(unique_risk_snps)}")

#Iterate through traits
for trait in set(disease_to_efo.values()):
    print(f"Trait: {trait}")
    
    traits = [key for key, value in disease_to_efo.items() if value == trait]

    # Get rsSNPs associated with a trait
    risk_snps = get_risk_snps(gwas_catalog, trait)
    print(risk_snps)
        
    for index, row in risk_snps.iterrows():
        rsSNP = row['SNPS']
        print(rsSNP)

        # Get information about a rssnp 
        snp_details = extract_snp_details(gwas_catalog, rsSNP, trait)
        
        if snp_details:
            print(snp_details)
            
            summary_stats = get_summary_stats_for_snp(snp_details, trait)
            print(summary_stats)
            
            if summary_stats:

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


