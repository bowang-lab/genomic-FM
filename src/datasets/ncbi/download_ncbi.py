import os
import subprocess

def create_taxid_species_map(filepath):
    """
    Create a map from taxid to species name from the 'nodes.dmp' file.
    Parameters:
    - filepath: Path to the 'nodes.dmp' file.
    Returns:
    - A dictionary mapping taxids to species names.
    """
    species_to_taxid = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Split the line into components. Adjust based on file's actual format.
            parts = line.strip().split('\t|\t')
            if parts:
                taxid = parts[0]  # Assuming the first column is the taxid.
                species_name = parts[1]  # Assuming the second column is the scientific name.
                species_to_taxid[species_name] = taxid

    return species_to_taxid

def download_genome(species_name, accession, output_dir):
    """
    Calls a Bash script to download genome data for a given species.
    
    Parameters:
        species_name (str): The species name.
        accession (str): Accession of assembly.
        output_dir (str): The output directory for downloading the data.
    """
    script_path = "./download_genome.sh"  
    try:
        subprocess.run([script_path, species_name, accession, output_dir], check=True)
        print(f"Successfully downloaded genome data for {species_name}.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while downloading genome data: {e}")
