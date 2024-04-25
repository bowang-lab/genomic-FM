import vcf
import pandas as pd
import gzip
import ftplib
from kipoiseq import Interval
import glob
from tqdm import tqdm
from src.sequence_extractor import RandomSequenceExtractor, FastaStringExtractor
from src.blast_search import run_blast_query
from src.datasets.ncbi_reference_genome.get_accession import search_species
import random
import os

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

def download_regulatory_gff(out_dir='./root/data/regulatory_features'):
    """Download regulatory features GFF for a given species"""
    for species in ensembl_regulatory_species:
        print(f"Downloading regulatory regions for {species}")

        ftp_url = 'ftp.ensembl.org'
        base_path = '/pub/current_regulation/'
        species_path = species.lower().replace(' ', '_')

        if os.path.exists(out_dir):
            return

        os.makedirs(out_dir, exist_ok=True)

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
                local_filename = os.path.join(out_dir, filename)
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

def get_regulatory_regions(sequence_length=1024, limit=None):
    feature_types = ["enhancer", "TF_binding_site", "CTCF_binding_site", "open_chromatin_region"]
    combined_data = []

    for species in ensembl_regulatory_species:
        print(f"Processing regulatory regions for {species}")
        tax_id = search_species(species)
        fasta_paths = glob.glob(os.path.join("./root/data", tax_id[0], "ncbi_dataset/data/GCF*/GCF*fna"))
        gtf_paths = glob.glob(os.path.join("./root/data", tax_id[0], "ncbi_dataset/data/GCF*/genomic.gtf"))

        if not fasta_paths or not gtf_paths:
            print(f"Required genomic files missing for {species}. Skipping...")
            continue

        fasta_extractor = FastaStringExtractor(fasta_paths[0])
        random_extractor = RandomSequenceExtractor(fasta_paths[0], gtf_paths[0])

        for feature_type in feature_types:
            features = get_feature_type(feature_type, species)
            print(f"Processing features of type {feature_type}:")

            feature_data = []

            for index, row in tqdm(features.iterrows()):
                if limit and index >= limit:
                    break

                chrom = "chr" + str(row['seqid'])
                start = int(row['start'])
                end = int(row['end'])
                interval = Interval(chrom, start, end).resize(sequence_length)

                # Extract the sequence
                sequence = fasta_extractor.extract(interval)
                feature_data.append((species, feature_type, sequence, interval))

            if feature_data:
                # Generate random sequences, passing intervals to avoid overlapping with known regions
                intervals = [data[-1] for data in feature_data]
                random_sequences = random_extractor.extract_random_sequence(
                    length_range=(sequence_length, sequence_length),
                    num_sequences=len(feature_data),
                    known_regions=intervals
                )

                combined_data.extend([(data[0], data[1], data[2], 1) for data in feature_data])
                combined_data.extend([(data[0], data[1], rand_seq, 0) for data, rand_seq in zip(feature_data, random_sequences)])

    return combined_data