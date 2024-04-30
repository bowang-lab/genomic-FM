import vcf
import pandas as pd
import gzip
from kipoiseq import Interval
import glob
from tqdm import tqdm
from src.sequence_extractor import RandomSequenceExtractor, FastaStringExtractor
from src.blast_search import run_blast_query
from src.datasets.ncbi_reference_genome.get_accession import search_species, get_chromosome_name
import random
import os
import requests

ensembl_regulatory_species = [
    "Oncorhynchus mykiss",
    "Salmo salar",
    "Scophthalmus maximus",
    "Sus scrofa",
    "Cyprinus carpio carpio",
    "Dicentrarchus labrax",
    "Gallus gallus",
    "Mus musculus",
    "Homo sapiens",
]

def download_regulatory_gff(out_dir='./root/data/regulatory_features'):
    """Download regulatory features GFF for a given species using specific URLs."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    species_urls = {
        "Cyprinus carpio carpio": "https://ftp.ensembl.org/pub/current_regulation/cyprinus_carpio_carpio/Cypcar_WagV4.0/cyprinus_carpio_carpio.Cypcar_WagV4.0.regulatory_features.20230913.gff.gz",
        "Dicentrarchus labrax": "https://ftp.ensembl.org/pub/current_regulation/dicentrarchus_labrax/dlabrax2021/dicentrarchus_labrax.dlabrax2021.regulatory_features.20230918.gff.gz",
        "Gallus gallus": "https://ftp.ensembl.org/pub/current_regulation/gallus_gallus/GRCg7b/gallus_gallus.bGalGal1.mat.broiler.GRCg7b.regulatory_features.20230815.gff.gz",
        "Homo sapiens": "https://ftp.ensembl.org/pub/current_regulation/homo_sapiens/homo_sapiens.GRCh38.Regulatory_Build.regulatory_features.20221007.gff.gz",
        "Mus musculus": "https://ftp.ensembl.org/pub/current_regulation/mus_musculus/mus_musculus.GRCm39.Regulatory_Build.regulatory_features.20221007.gff.gz",
        "Scophthalmus maximus": "https://ftp.ensembl.org/pub/current_regulation/scophthalmus_maximus/ASM1334776v1/scophthalmus_maximus.ASM1334776v1.regulatory_features.20230920.gff.gz",
        "Sus scrofa": "https://ftp.ensembl.org/pub/current_regulation/sus_scrofa/Sscrofa11.1/sus_scrofa.Sscrofa11.1.regulatory_features.20230814.gff.gz",
        "Salmo salar": "https://ftp.ensembl.org/pub/current_regulation/salmo_salar/Ssal_v3.1/salmo_salar.Ssal_v3.1.regulatory_features.20230906.gff.gz",
        "Oncorhynchus mykiss": "https://ftp.ensembl.org/pub/current_regulation/oncorhynchus_mykiss/USDA_OmykA_1.1/oncorhynchus_mykiss.USDA_OmykA_1.1.regulatory_features.20230919.gff.gz"
    }

    for species, url in species_urls.items():
        filename = url.split('/')[-1]
        local_filename = os.path.join(out_dir, filename)

        print(f"Downloading regulatory regions for {species}: {filename}")

        if os.path.exists(local_filename):
            print(f"{filename} already downloaded. Skipping.")
            continue

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Downloaded {filename} to {local_filename}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}. Error: {e}")

    return out_dir

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

def get_eukaryote_regulatory(sequence_length=1024, limit=None):
    feature_types = ["enhancer", "TF_binding_site", "CTCF_binding_site", "open_chromatin_region"]
    combined_data = []

    for species in ensembl_regulatory_species:
        print(f"Processing regulatory regions for {species}")
        tax_id = search_species(species)
        fasta_paths = glob.glob(os.path.join("./root/data", tax_id[0], "ncbi_dataset/data/GC*/GC*fna"))

        if not fasta_paths :
            print(f"Required genomic fasta file is missing for {species}. Skipping...")
            continue

        fasta_extractor = FastaStringExtractor(fasta_paths[0])
        ncbi_ids = [key for key in fasta_extractor.fasta.keys() if key.startswith(('NC', 'NL'))]
        ncbi_chromosome_mapping = {}

        for ncbi_id in ncbi_ids:
            chromosome_name = get_chromosome_name(ncbi_id)
            ncbi_chromosome_mapping[ncbi_id] = chromosome_name
        chromosome_ncbi_mapping = {value: key for key, value in ncbi_chromosome_mapping.items()}

        random_extractor = RandomSequenceExtractor(fasta_paths[0])

        for feature_type in feature_types:
            features = get_feature_type(feature_type, species)
            print(f"Processing {len(features)} features of type {feature_type}:")

            feature_data = []
            for index, row in tqdm(features.iterrows()):
                if limit and index >= limit:
                    break

                if row['seqid'] not in chromosome_ncbi_mapping.keys():
                    continue 
                
                chrom = chromosome_ncbi_mapping[row['seqid']]
                start = int(row['start'])
                end = int(row['end'])
                interval = Interval(chrom, start, end).resize(sequence_length)

                # Extract the sequence
                sequence = fasta_extractor.extract(interval)
                feature_data.append([species, feature_type, sequence, interval])

            if feature_data:
                # Generate random sequences, passing intervals to avoid overlapping with known regions
                intervals = [data[-1] for data in feature_data]
                random_sequences = random_extractor.extract_random_sequence(
                    length_range=(sequence_length, sequence_length),
                    num_sequences=len(feature_data),
                    known_regions=intervals
                )

                combined_data.extend([[[data[0], data[1], data[2]], 1] for data in feature_data])
                combined_data.extend([[[data[0], data[1], rand_seq], 0] for data, rand_seq in zip(feature_data, random_sequences)])

    return combined_data