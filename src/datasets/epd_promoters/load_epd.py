import json
import os
import requests
import zipfile
import pandas as pd
import os
from tqdm import tqdm
import glob
from src.sequence_extractor import RandomSequenceExtractor, FastaStringExtractor
from src.blast_search import run_blast_query
from src.datasets.ncbi_reference_genome.get_accession import search_species
import random

ppd_species = [
    "Acinetobacter baumannii ATCC 17978",
    "Agrobacterium tumefaciens str C58",
    "Bacillus subtilis subsp. subtilis str. 168",
    "Bradyrhizobium japonicum USDA 110",
    "Burkholderia cenocepacia J2315",
    "Campylobacter jejuni 81-176",
    "Campylobacter jejuni 81116",
    "Campylobacter jejuni NCTC11168",
    "Campylobacter jejuni RM1221",
    "Corynebacterium diphtheriae NCTC 13129",
    "Corynebacterium glutamicum ATCC 13032",
    "Escherichia coli str K-12 substr. MG1655",
    "Haloferax volcanii DS2",
    "Helicobacter pylori strain 26695",
    "Klebsiella aerogenes KCTC 2190",
    "Nostoc sp. PCC7120",
    "Onion yellows phytoplasma OY-M",
    "Paenibacillus riograndensis SBR5",
    "Pseudomonas putida strain KT2440",
    "Shigella flexneri 5a str. M90T",
    "Sinorhizobium meliloti 1021",
    "Staphylococcus aureus MW2",
    "Staphylococcus epidermidis ATCC 12228",
    "Streptococcus pyogenes strain S119",
    "Synechococcus elongatus PCC 7942",
    "Synechocystis sp. PCC 6803",
    "Thermococcus kodakarensis KOD1",
    "Xanthomonas campestris pv. campestrie B100",
    "other species"
]

def download_ppd(directory="./root/data/ppd"):
    """
    Downloads a ZIP file from a given URL and extracts its contents into a specified directory.

    Args:
    - url (str): The URL of the ZIP file to download.
    - directory (str, optional): The directory where the ZIP file will be extracted. Defaults to "./root/data".

    Returns:
    - None
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    zip_path = os.path.join(directory, "ppd.zip")
    response = requests.get("http://lin-group.cn/database/ppd/todownload/csv.zip")
    with open(zip_path, "wb") as zip_file:
        zip_file.write(response.content)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(directory)

    os.remove(zip_path)
    print(f"Downloaded and extracted files to {directory}")

def get_promoter_prokaryote(species, directory='./root/data/ppd/csv'):
    """
    Reads all CSV files in a given directory that match the specified species name
    and extracts the "GeneName", "PromoterSeq", "TSSPosition", and "Strand" columns,
    then converts the data into JSON format.

    Args:
    - directory (str): The directory containing the CSV files.
    - species_name (str): The name of the species to filter the CSV files.

    Returns:
    - str: A JSON string containing the extracted data from all matching CSV files.
    """

    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv') and species in f]
    all_data = []
    for file_name in csv_files:
        file_path = os.path.join(directory, file_name)
        df = pd.read_csv(file_path)
        extracted_data = df[['GeneName', 'PromoterSeq', 'TSSPosition', 'Strand']]
        extracted_data.loc[extracted_data.index, 'Species'] = species
        all_data.extend(extracted_data.to_dict('records'))
    return all_data

def get_prokaryote_promoters(sequence_length=1024, limit=None):
    combined_tuples = []

    for prokaryote_species in ppd_species:
        print(f"Processing {prokaryote_species}")
        tax_id = search_species(prokaryote_species)
        if not tax_id:
            print(f"Tax ID not found for {prokaryote_species}. Skipping...")
            continue

        data_path = os.path.join("./root/data", tax_id[0], "ncbi_dataset/data/GCF*")
        fasta_file_path = glob.glob(f"{data_path}/GCF*fna")
        gtf_file_path = glob.glob(f"{data_path}/genomic.gtf")
        if not fasta_file_path or not gtf_file_path:
            print(f"No genomic files found for {prokaryote_species} in {data_path}. Skipping...")
            continue

        fasta_extractor = FastaStringExtractor(fasta_file_path[0])
        promoters = get_promoter_prokaryote(prokaryote_species)
        promoter_data = []

        for index, promoter in enumerate(tqdm(promoters, desc=f"Extracting promoters for {prokaryote_species}")):
            if limit and index >= limit:
                break
            species, gene_name, sequence = promoter['Species'], promoter['GeneName'], promoter['PromoterSeq'].upper()

            if sequence:
                interval = run_blast_query(sequence, fasta_file_path[0])
                if interval is not None:
                    interval = interval.resize(sequence_length)
                    extended_sequence = fasta_extractor.extract(interval)
                    promoter_data.append((species, gene_name, extended_sequence, interval))

        if promoter_data:
            intervals = [data[-1] for data in promoter_data] 
            random_sequences = RandomSequenceExtractor(fasta_file_path[0], gtf_file_path[0]).extract_random_sequence(
                length_range=(sequence_length, sequence_length),
                num_sequences=len(promoter_data),
                known_regions=intervals
            )
            combined_tuples.extend([(data[0], data[1], data[2], 1) for data in promoter_data])
            combined_tuples.extend([(data[0], data[1], seq, 0) for data, seq in zip(promoter_data, random_sequences)])
        random.shuffle(combined_tuples)

    return combined_tuples
