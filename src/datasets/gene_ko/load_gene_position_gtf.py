from kipoiseq.dataloaders.sequence import AnchoredGTFDl
from kipoiseq.extractors import FastaStringExtractor
from kipoiseq.transforms.functional import resize_interval, one_hot_dna
import pyranges as pr
import os
import subprocess
import pyfaidx
import requests
import gzip

def download_fasata_file(fasta_file):
    if not os.path.exists(fasta_file):
        run_bash_commands_fasta(fasta_file)

def download_gft_file(gft_file):
    if not os.path.exists(gft_file):
        run_bash_commands_gtf(gft_file)

def run_bash_commands_fasta(fasta_file):
    try:
        # Create directory
        os.makedirs('./root/data', exist_ok=True)

        # Download the file
        print("Downloading the hg38 file...")
        url = 'http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
        response = requests.get(url)
        compressed_file = './root/data/hg38.fa.gz'
        with open(compressed_file, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

        # Decompress the file
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(fasta_file, 'wb') as f_out:
                f_out.write(f_in.read())
        print("Decompression complete.")

        # Index the file with pyfaidx
        pyfaidx.Faidx(fasta_file)
        print("Indexing complete.")

        # List contents of the directory
        subprocess.run(['ls', './root/data'])
    except Exception as e:
        print(f"An error occurred: {e}")


def run_bash_commands_gtf(gtf_file):
    try:
        # Create directory
        os.makedirs('./root/data', exist_ok=True)

        # Download the file
        print("Downloading the GTF file...")
        url = 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz'
        response = requests.get(url)
        compressed_file = './root/data/gencode.v45.annotation.gtf.gz'
        with open(compressed_file, 'wb') as f:
            f.write(response.content)
        print("Download complete.")

        # Decompress the file
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(gtf_file, 'wb') as f_out:
                f_out.write(f_in.read())
        print("Decompression complete.")

        # List contents of the directory
        subprocess.run(['ls', './root/data'])
    except Exception as e:
        print(f"An error occurred: {e}")


class CustomAnchoredGTFDl(AnchoredGTFDl):
    def __init__(self,
                 num_upstream, num_downstream,
                 gtf_file="root/data/gencode.v45.annotation.gtf",
                 fasta_file="root/data/hg38.fa",
                 anchor='tss',
                 transform=None,
                 interval_attrs=["gene_name", "Strand", "anchor_pos", "Start", "End"],
                 use_strand=True):
        # download fasta file
        download_fasata_file(fasta_file)
        # download gtf file
        download_gft_file(gtf_file)


        # Read GTF without filtering
        gtf = pr.read_gtf(gtf_file).df

        # Initialize anchor
        if isinstance(anchor, str):
            anchor = anchor.lower()
            if anchor in self._function_mapping:
                anchor = self._function_mapping[anchor]
            else:
                raise Exception("No valid anchorpoint was chosen")
        self._gtf_anchor = anchor(gtf)

        # Other parameters
        self._use_strand = use_strand
        self._fa = FastaStringExtractor(fasta_file, use_strand=self._use_strand)
        self._transform = transform if transform else lambda x: x
        self._num_upstream = num_upstream
        self._num_downstream = num_downstream
        self._interval_attrs = interval_attrs

    def filter_by_gene(self, gene_names):
        if isinstance(gene_names, str):
            gene_names = [gene_names]
        self._gtf_anchor = self._gtf_anchor[self._gtf_anchor['gene_name'].isin(gene_names)]
