import os
import subprocess

def download_species(outdir):
    """
    Downloads the NCBI Taxonomy dump and extracts it to a specified directory.

    Parameters:
    - outdir: The directory where the taxdump.tar.gz will be downloaded and extracted.
    """
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(f'{outdir}/nodes.dmp'):
        url = "https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz"
        tar_gz_path = os.path.join(outdir, "taxdump.tar.gz")

        # Download the tar.gz file using wget
        download_cmd = f"wget {url} -O {tar_gz_path}"
        subprocess.run(download_cmd, shell=True, check=True)

        # Extract the downloaded tar.gz file
        extract_cmd = f"tar -xvzf {tar_gz_path} -C {outdir}"
        subprocess.run(extract_cmd, shell=True, check=True)

        os.remove(tar_gz_path)


def create_taxid_species_map():
    """
    Create a map from taxid to species name from the 'nodes.dmp' file.
    Returns:
    - A dictionary mapping taxids to species names.
    """
    species_to_taxid = {}
    with open('./root/data/nodes.dmp', 'r') as file:
        for line in file:
            # Split the line into components. Adjust based on file's actual format.
            parts = line.strip().split('\t|\t')
            if parts:
                taxid = parts[0]  # Assuming the first column is the taxid.
                species_name = parts[1]  # Assuming the second column is the scientific name.
                species_to_taxid[species_name] = taxid

    return species_to_taxid

def download_species_genome(species, accession, outdir):
    """
    Downloads genome data using NCBI's datasets tool, unzips the downloaded file,
    and then rehydrates it.

    Parameters:
    - accession: The accession number of the genome to download.
    - outdir: The output directory where the downloaded files will be stored.
    - species: The species name, used to create subdirectories within the output directory.
    """
    # Construct the filename and directory paths
    filename = f"{outdir}/{species}/{accession}.zip"
    directory = f"{outdir}/{species}"

    if not os.path.exists(directory):
        os.mkdir(directory)

    # Download the genome data
    download_cmd = (
        f"datasets download genome accession {accession} --dehydrated "
        f"--include genome,rna,cds,protein,gtf --filename {filename}"
    )
    subprocess.run(download_cmd, shell=True, check=True)

    # Unzip the downloaded file
    unzip_cmd = f"unzip -o {filename} -d {directory}"
    subprocess.run(unzip_cmd, shell=True, check=True)

    # Rehydrate the dataset
    rehydrate_cmd = f"datasets rehydrate --directory {directory}"
    subprocess.run(rehydrate_cmd, shell=True, check=True)

