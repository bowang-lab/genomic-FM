import os
import argparse
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.ncbi_reference_genome.get_accession import get_main_assembly_accession, search_species, fetch_species_details
from src.datasets.ncbi_reference_genome.download_ncbi import create_species_taxid_map, download_species_genome, download_species, download_gene_species


# Setup argparse
parser = argparse.ArgumentParser(description="Download genome data for a given species.")
parser.add_argument("--outdir", default = './root/data',  help="The output directory to save the data.")
parser.add_argument("--species", default=None, help="The species to download data for.")
parser.add_argument("--gene", default=None, help="The gene symbol to process data for.")
parser.add_argument("--archaea", action='store_true', help="Download only archaea.")
parser.add_argument("--bacteria", action='store_true', help="Download only bacteria.")
parser.add_argument("--eukaryotes", action='store_true', help="Download only eukaryotes.")
parser.add_argument("--complete", action='store_true', help="Download only complete genomes.")
parser.add_argument("--get_latest", action='store_true', help="Retrieve latest accession for species.")
parser.add_argument("--get_reference", action='store_true', help="Retrieve reference genome accession for species.")

args = parser.parse_args()

if not os.path.exists(os.path.join(args.outdir,"taxdump.tar.gz")):
    download_species(args.outdir)

if args.species and args.gene:
    download_gene_species(args.gene, args.species, args.outdir)
elif args.species:
    if args.get_reference:
        accession = get_main_assembly_accession(args.species)
    elif args.get_latest:
        accession = get_main_assembly_accession(args.species, reference=False)
    else:
        raise ValueError("Please specify accession type --get_latest or --get_reference")

    tax_id = search_species(args.species)
    print(f"NCBI Taxonomy IDs for {args.species}: {tax_id}")

    if isinstance(tax_id, list):
        tax_id = tax_id[0]

    try:
        print(f"Details for Taxonomy ID {tax_id}:")
        details = fetch_species_details(tax_id)
        print(details)
        download_species_genome(tax_id, accession, args.outdir)
    except Exception as e:
        print(f"Failed to fetch sequence for Species {args.species}: {e}")

else:
    species_to_taxids = create_species_taxid_map()
    species_list = list(species_to_taxids.keys())

    if args.eukaryotes:
        eukaryotes = pd.read_table("./root/data/ncbi_reference_eukaryotes.txt",sep='\t')
        eukaryotes_list = list(eukaryotes['Scientific.species'].str.rstrip())
        species_list = [species for species in species_list if species in eukaryotes_list]
        print(f"Number of eukaryotic species: {len(species_list)}")

       # Eukaryotes
        eukaryotes = pd.read_table("./root/data/eukaryotes_refseq_summary.csv",sep=',')
        if args.complete:
            eukaryotes = eukaryotes[eukaryotes['Level'] == 'Complete']
        
       # List species
        species_list = list(eukaryotes['#Organism Name'].str.rstrip())
        print(f"Number of eukaryotic species: {len(species_list)}")

    if args.archaea:
        # Archaea
        archaea = pd.read_table("./root/data/archaea_refseq_summary.csv",sep=',')
        if args.complete:
            archaea = archaea[archaea['Level'] == 'Complete']
        
        # List species
        species_list = list(archaea['#Organism Name'].str.rstrip())
        print(f"Number of archaeaic species: {len(species_list)}")

    if args.bacteria:
        # Bacteria
        bacteria = pd.read_table("./root/data/bacteria_refseq_summary.csv",sep=',')
        if args.complete:
            bacteria = bacteria[bacteria['Level'] == 'Complete']
        
        # List species
        species_list = list(bacteria['#Organism Name'].str.rstrip())
        print(f"Number of bacteria species: {len(species_list)}")

    for species in species_list:
        try:
            if args.get_reference:
                accession = get_main_assembly_accession(species)
            elif args.get_latest:
                accession = get_main_assembly_accession(species, reference=False)
            else: 
                raise ValueError("Please specify accession type --get_latest or --get_reference")

            tax_id = search_species(species)
            print(f"{species}: {accession}")

        except Exception as e:
            print(f"Failed to fetch details for Species {species}: {e}")
            continue

        if isinstance(tax_id, str):
            if os.path.exists(os.path.join(args.outdir, tax_id)):
                continue
        elif isinstance(tax_id, list):
            if not tax_id or os.path.exists(os.path.join(args.outdir, tax_id[0])):
                continue
            tax_id = tax_id[0]

        try:
            print(f"Details for Taxonomy ID {tax_id}:")
            details = fetch_species_details(tax_id)
            print(details)
            download_species_genome(tax_id, accession, args.outdir)
        except Exception as e:
            print(f"Failed to fetch sequence for Species {species}: {e}")
