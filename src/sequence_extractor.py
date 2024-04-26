import os
import subprocess
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import requests
import gzip
import random

class RandomSequenceExtractor:
    def __init__(self, fasta_file, gtf_file):
        self.fasta_file = fasta_file

    def extract_random_sequence(self, length_range=(200, 1000), num_sequences=10, known_regions=None):
        fasta = pyfaidx.Fasta(self.fasta_file)
        selected_sequences = []

        for _ in range(num_sequences):
            chrom = random.choice(list(fasta.keys()))
            chrom_length = len(fasta[chrom])

            # Ensure the random sequence does not overlap with known promoters
            is_known_region = True
            while is_known_region:
                start = random.randint(1, chrom_length - length_range[1])  # Adjust start point to allow enough space for maximum length
                length = random.randint(*length_range)
                end = start + length
                is_known_region = any(
                    feature.start <= start <= feature.end or feature.start <= end <= feature.end
                    for feature in known_regions if feature.chrom == chrom
                )

            sequence = fasta[chrom][start:end].seq.upper()
            selected_sequences.append(sequence)

        return selected_sequences
    
class GenomeSequenceExtractor:
    def __init__(self, fasta_file='./root/data/hg38.fa'):
        self.fasta_file = fasta_file
        if not os.path.exists(fasta_file):
            self.run_bash_commands()
        self.fasta_extractor = FastaStringExtractor(fasta_file)

    def run_bash_commands(self):
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
                with open(self.fasta_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            print("Decompression complete.")

            # Index the file with pyfaidx
            pyfaidx.Faidx(self.fasta_file)
            print("Indexing complete.")

            # List contents of the directory
            subprocess.run(['ls', './root/data'])
        except Exception as e:
            print(f"An error occurred: {e}")

    def extract_sequence(self, variant, sequence_length=1024):
        interval = kipoiseq.Interval(variant.chrom, variant.start, variant.start).resize(sequence_length)
        seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=self.fasta_extractor)
        center = interval.center() - interval.start
        if self.fasta_extractor.is_valid_chromosome(interval.chrom) is None:
            return None, None
        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, [variant], anchor=center)
        return reference, alternate

    def extract_sequence_from_record(self, record, sequence_length=1024):
        # Extract information from the record
        chr = record['Chromosome']
        pos = record['Position']
        ref = record['Reference Base']
        alt = record['Alternate Base'][0]
        variant_id = record['ID']
        if "chr" in chr:
            chr = chr[3:]

        # Create a Variant object using kipoiseq, which represents the genetic variant
        variant = kipoiseq.Variant(f"chr{chr}", pos, ref, alt, id=f'rs{variant_id}')
        # Extract the reference and alternate sequences surrounding the variant
        return self.extract_sequence(variant, sequence_length=sequence_length)

class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}
        synonynms_file = './root/data/chromAliases.txt'
        if not os.path.exists(synonynms_file):
            self.download_synonyms()
        # mapping the NCBI chromosome names to UCSC chromosome names if necessary
        self.chrom_synonyms = self.read_synonyms(synonynms_file)

    def download_synonyms(self,url="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/chromAlias.txt.gz"):
        print("Downloading the chromAlias file...")
        response = requests.get(url)
        compressed_file = './root/data/chromAlias.txt.gz'
        with open(compressed_file, 'wb') as f:
            f.write(response.content)
        with gzip.open(compressed_file, 'rb') as f_in:
            with open('./root/data/chromAliases.txt', 'wb') as f_out:
                f_out.write(f_in.read())

    def read_synonyms(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()
        synonyms = {}
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.split()
            synonyms[parts[0]] = parts[1]
        return synonyms

    def is_valid_chromosome(self, chrom_name):
        if chrom_name not in self.fasta:
            if chrom_name[3:] in self.chrom_synonyms:
                chrom_name = self.chrom_synonyms[chrom_name[3:]]
                if chrom_name not in self.fasta:
                    print(f"Chromosome {chrom_name} name exists in the synonyms file but"
                          "not found in the reference genome.")
                    return None
            else:
                print(f"Chromosome {chrom_name} not found in the reference genome.")
                return None
        return chrom_name

    def extract(self, interval: Interval, sequence_length = None, **kwargs) -> str:
        if sequence_length is not None:
            interval = interval.resize(sequence_length)
        chrom_name = interval.chrom
        chrom_name = self.is_valid_chromosome(chrom_name)
        chromosome_length = self._chromosome_sizes[chrom_name]
        trimmed_interval = Interval(chrom_name, max(interval.start, 0), min(interval.end, chromosome_length))
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom, trimmed_interval.start + 1, trimmed_interval.stop).seq).upper()
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()
