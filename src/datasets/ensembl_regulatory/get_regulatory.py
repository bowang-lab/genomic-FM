import vcf
import pandas as pd
import gzip
import ftplib
import os
import glob

ensembl_regulatory_species = [
    "Cyprinus carpio carpio",
    "Dicentrarchus labrax",
    "Gallus gallus",
    "Homo sapiens",
    "Mus musculus",
    "Oncorhynchus mykiss",
    "Salmo salar",
    "Scophthalmus maximus",
    "Sus scrofa"
]

def download_regulatory_gff(species, target_directory = './root/data/regulatory_features'):
    """Download regulatory features GFF for a given species"""
    if species not in ensembl_regulatory_species:
        print(f"Regulatory data for {species_name} does not exist in Ensembl.")
        return

    ftp_url = 'ftp.ensembl.org'
    base_path = '/pub/current_regulation/'
    species_path = species.lower().replace(' ', '_')

    if os.path.exists(target_directory):
        return

    os.makedirs(target_directory, exist_ok=True)

    with ftplib.FTP(ftp_url) as ftp:
        ftp.login()  # Anonymous login

        full_path = os.path.join(base_path, species_path)
        try:
            ftp.cwd(full_path)
        except ftplib.error_perm as e:
            print(f"Could not access directory for {species}. Error: {e}")
            return

        files = ftp.nlst()  # List all files in the species directory
        gff_files = [f for f in files if f.endswith('.gff.gz')]

        if not gff_files:
            print(f"No regulatory GFF files found for {species}.")
            return

        for filename in gff_files:
            local_filename = os.path.join(target_directory, filename)
            with open(local_filename, 'wb') as file:
                ftp.retrbinary('RETR ' + filename, file.write)
                print(f"Downloaded {filename} to {local_filename}")
        else:
            print(f"Downloaded all regulatory GFF files for {species}.")

def parse_attributes(attribute_string):
    """Parse the GFF attributes column into a dictionary."""
    attributes = attribute_string.strip().split(";")
    attr_dict = {}
    for attribute in attributes:
        if "=" in attribute:
            key, value = attribute.split("=", 1)
            if key == "ID" and ":" in value:
                _, value = value.split(":", 1)
            attr_dict[key] = value
    return attr_dict

def get_feature_type(filter_type, species,target_directory = './root/data/regulatory_features'):
    """Read and parse GFF file"""
    species_path = species.lower().replace(' ', '_')+"*"
    filepath = glob.glob(os.path.join(target_directory,species_path))[0]

    col_names = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']

    with gzip.open(filepath, 'rt') as file:
        df = pd.read_csv(file, sep='\t', comment='#', names=col_names, dtype={"attributes": str}, low_memory=False)

    attributes_df = df['attributes'].apply(parse_attributes).apply(pd.Series)
    reg_data = pd.concat([df.drop(columns=['attributes']), attributes_df], axis=1)
    filtered_data = reg_data[reg_data['type'] == filter_type]
    return filtered_data

