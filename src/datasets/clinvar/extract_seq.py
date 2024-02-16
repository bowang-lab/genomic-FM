import os
import subprocess
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import requests
import gzip


def run_bash_commands(fasta_file):
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


class FastaStringExtractor:

    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom, max(interval.start, 0), min(interval.end, chromosome_length))
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom, trimmed_interval.start + 1, trimmed_interval.stop).seq).upper()
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

fasta_file = './root/data/hg38.fa'
if not os.path.exists(fasta_file):
    run_bash_commands(fasta_file)
fasta_extractor = FastaStringExtractor(fasta_file)

def extract_sequence(variant, sequence_length=1024):
    interval = kipoiseq.Interval(variant.chrom, variant.start, variant.start).resize(sequence_length)
    seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=fasta_extractor)
    center = interval.center() - interval.start
    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)
    return reference, alternate

# ---------------------------------
# ### Example usage ###
# SEQUENCE_LENGTH = 1024
# variant = kipoiseq.Variant('chr16', 57025062, 'C', 'T', id='rs11644125')
# reference, alternate = extract_sequence(fasta_file, variant, sequence_length=SEQUENCE_LENGTH)
# print(f"Reference sequence: {reference}")
# print(f"Alternate sequence: {alternate}")
# fasta_extractor.close()
# ---------------------------------
