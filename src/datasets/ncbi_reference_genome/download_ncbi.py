import os
import subprocess

def download_gene_species(gene_symbol, species, outdir):
    """
    Fetch and process genetic information for a specified gene symbol and species.
    
    Parameters:
    gene_symbol (str): The gene symbol to query.
    species (str): The species of interest.
    outdir (str): The output directory to store results.
    """
    # Create a unique directory for the gene
    gene_dir = os.path.join(outdir, f"ncbi_dataset/data/{gene_symbol}")
    os.makedirs(gene_dir, exist_ok=True)
    
    # Fetch Orthologous Information
    print(f"Fetching orthologous information for {gene_symbol} from the species {species}...")
    orthologs_file = os.path.join(gene_dir, f"{gene_symbol}_{species}_orthologs.jsonl")
    subprocess.run(["datasets", "summary", "gene", "symbol", gene_symbol, "--ortholog", species, 
                    "--report", "product", "--as-json-lines"], stdout=open(orthologs_file, "w"))
    
    # Generate a summary for the provided gene symbol
    print(f"Generating summary for {gene_symbol}...")
    summary_file = os.path.join(gene_dir, f"{gene_symbol}_{species}_summary.txt")
    subprocess.run(["datasets", "summary", "gene", "symbol", gene_symbol], stdout=open(summary_file, "w"))
    
    # Convert JSONL to TSV
    print("Converting JSONL to TSV...")
    tsv_file = os.path.join(gene_dir, f"{gene_symbol}_{species}_transcript_cds_protein.tsv")
    subprocess.run(["dataformat", "tsv", "gene-product", "--inputfile", orthologs_file, "--fields", 
                    "gene-id,gene-type,tax-name,symbol,transcript-accession,transcript-length,"\
                    "transcript-cds-accession,transcript-ensembl-transcript,transcript-genomic-location-accession,"\
                    "transcript-transcript-type,transcript-protein-accession,transcript-protein-ensembl-protein,"\
                    "transcript-protein-name,transcript-protein-length,transcript-protein-isoform,"\
                    "transcript-protein-mat-peptide-accession,transcript-protein-mat-peptide-name,"\
                    "transcript-protein-mat-peptide-length"], stdout=open(tsv_file, "w"))
    
    # Identify Longest Transcripts and Their Proteins
    print("Identifying longest transcripts...")
    longest_list_file = os.path.join(gene_dir, f"{gene_symbol}_{species}_longest.list")
    with open(tsv_file) as f, open(longest_list_file, "w") as out_f:
        next(f)  # Skip header
        gene_ids = set(line.split("\t")[0] for line in f)
        for gene_id in gene_ids:
            f.seek(0)
            next(f)  # Skip header again for each gene_id
            longest_transcript = max((line for line in f if line.startswith(gene_id)), 
                                     key=lambda x: int(x.split("\t")[13]))
            out_f.write("\n".join(longest_transcript.split("\t")[8:11]) + "\n")
    
    # Download Sequences
    print("Downloading sequences...")
    zip_file = os.path.join(gene_dir, f"{gene_symbol}_{species}_longest.zip")
    subprocess.run(["datasets", "download", "gene", "accession", "--inputfile", longest_list_file, 
                    "--fasta-filter-file", longest_list_file, "--filename", zip_file])
    
    # Unzip the downloaded file
    print("Unzipping sequences...")
    subprocess.run(["unzip", zip_file, "-d", gene_dir])
    
    print(f"Process completed for {gene_symbol}.")

def download_species(outdir):
    """
    Downloads the NCBI Taxonomy dump and extracts it to a specified directory.

    Parameters:
    - outdir: The directory where the taxdump.tar.gz will be downloaded and extracted.
    """
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(f'{outdir}/nodes.dmp'):
        url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
        tar_gz_path = os.path.join(outdir, "taxdump.tar.gz")

        # Download the tar.gz file using wget
        download_cmd = f"wget {url} -O {tar_gz_path}"
        subprocess.run(download_cmd, shell=True, check=True)

        # Extract the downloaded tar.gz file
        extract_cmd = f"tar -xvzf {tar_gz_path} -C {outdir}"
        subprocess.run(extract_cmd, shell=True, check=True)

        os.remove(tar_gz_path)

def create_species_taxid_map():
    """
    Create a map from taxid to species name from the 'nodes.dmp' file.
    Returns:
    - A dictionary mapping taxids to species names.
    """
    species_to_taxid = {}
    with open('./root/data/names.dmp', 'r') as file:
        for line in file:
            # Split the line into components. Adjust based on file's actual format.
            parts = line.strip().split('\t|\t')
            if parts:
                taxid = parts[0]  # Assuming the first column is the taxid.
                species_name = parts[1]  # Assuming the second column is the scientific name.
                if taxid not in ["1","2"]:
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
        f"--include gtf,rna,protein,genome,seq-report --filename {filename}"
    )
    subprocess.run(download_cmd, shell=True, check=True)

    # Unzip the downloaded file
    unzip_cmd = f"unzip -o {filename} -d {directory}"
    subprocess.run(unzip_cmd, shell=True, check=True)

    # Rehydrate the dataset
    rehydrate_cmd = f"datasets rehydrate --directory {directory}"
    subprocess.run(rehydrate_cmd, shell=True, check=True)
