import os
import argparse
from src.datasets.ncbi_reference_genome.get_accession import get_main_assembly_accession, search_species, fetch_species_details
from src.datasets.ncbi_reference_genome.download_ncbi import create_species_taxid_map, download_species_genome, download_species, download_gene_species

# Setup argparse
parser = argparse.ArgumentParser(description="Download genome data for a given species.")
parser.add_argument("--outdir", default = './root/data',  help="The output directory to save the data.")
parser.add_argument("--species", default=None, help="The species to download data for.")
parser.add_argument("--gene", default=None, help="The gene symbol to process data for.")
parser.add_argument("--get_latest", action='store_true', help="Retrieve latest accession for species.")
parser.add_argument("--get_reference", action='store_true', help="Retrieve reference genome accession for species.")
args = parser.parse_args()

download_species(args.outdir)

if args.species and args.gene:
    download_gene_species(args.gene, args.species, args.outdir)
elif args.species:
    if args.get_reference:
        accession = get_main_assembly_accession(args.species)
    else:
        accession = get_main_assembly_accession(args.species, reference=False)

    print(f"{args.species}: {accession}")

    tax_id = search_species(args.species)
    print(f"NCBI Taxonomy IDs for {args.species}: {tax_id}")

    if tax_id:
        details = fetch_species_details(tax_id)
        print(f"Details for Taxonomy ID {tax_id}:")
        print(details)
        print(accession)
        download_species_genome(tax_id[0], accession, args.outdir)
    else:
        print("Taxid for the specified species could not be found.")
else:
    taxid_to_species = create_species_taxid_map()
    species_list = list(taxid_to_species.keys())
    for species in species_list:
        try:
            if args.get_reference:
                accession = get_main_assembly_accession(species)
            else:
                accession = get_main_assembly_accession(species, reference=False)
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
