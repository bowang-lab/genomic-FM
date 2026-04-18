import os
import subprocess
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import requests
import gzip
import random
from .encoding_region_filter import is_encoding, load_gtf

class RandomSequenceExtractor:
    def __init__(self, fasta_file):
        self.fasta_file = fasta_file

    def extract_random_sequence(self, length_range=(200, 1000), num_sequences=10, known_regions=None):
        fasta = pyfaidx.Fasta(self.fasta_file)
        selected_sequences = []

        # Initialize known_regions to an empty list if None is provided
        if known_regions is None:
            known_regions = []

        for _ in range(num_sequences):
            chrom = random.choice(list(fasta.keys()))
            chrom_length = len(fasta[chrom])

            if chrom_length < length_range[0]:  # Check if chromosome is shorter than the minimum length
                sequence = fasta[chrom][:].seq.upper()  # Extract the whole sequence
                pad_length = length_range[1] - chrom_length
                padded_sequence = ('N' * (pad_length // 2)) + sequence + ('N' * ((pad_length + 1) // 2))
                selected_sequences.append(padded_sequence)
                continue

            # Randomly generate sequences ensuring they do not overlap with known regions
            is_known_region = True
            while is_known_region:
                start = random.randint(0, chrom_length - length_range[1])
                length = random.randint(*length_range)
                end = start + length - 1  # Adjust index for inclusive end
                is_known_region = any(
                    feature.start <= start <= feature.end or feature.start <= end <= feature.end
                    for feature in known_regions if feature.chrom == chrom
                )
                if not is_known_region:  # Only append sequence if it does not overlap with known regions
                    sequence = fasta[chrom][start:end + 1].seq.upper()  # Correct end index for slicing
                    selected_sequences.append(sequence)

        return selected_sequences

class GenomeSequenceExtractor:
    def __init__(self, fasta_file='./root/data/hg19.fa', encoding_region_filter=None):
        self.fasta_file = fasta_file
        if not os.path.exists(fasta_file):
            self.run_bash_commands()
        self.fasta_extractor = FastaStringExtractor(fasta_file)
        if encoding_region_filter is not None:
            self.GTF = load_gtf()
        else:
            self.GTF = None

    def run_bash_commands(self):
        try:
            # Create directory
            os.makedirs('./root/data', exist_ok=True)

            # Download the file
            print("Downloading the hg19 file...")
            url = 'http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz'
            response = requests.get(url)
            compressed_file = './root/data/hg19.fa.gz'
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

    def extract_multi_variant_sequence(self, variants: list, sequence_length: int = 1024):
        """
        Extract sequence with multiple variants applied.

        Uses kipoiseq's multi-variant support to apply all variants at once.
        Variants must be on the same chromosome and close enough to fit within
        the sequence_length window.

        Args:
            variants: List of dicts with keys 'chrom', 'pos', 'ref', 'alt', 'variant_id'
                      OR list of kipoiseq.Variant objects
            sequence_length: Length of sequence to extract (default 1024)

        Returns:
            Tuple of (reference_sequence, alternate_sequence) or (None, None) on error
        """
        if not variants:
            return None, None

        # Convert dicts to kipoiseq.Variant objects if needed
        variant_objects = []
        for v in variants:
            if isinstance(v, dict):
                chrom = v['chrom']
                if not chrom.startswith('chr'):
                    chrom = f"chr{chrom}"
                variant_obj = kipoiseq.Variant(
                    chrom,
                    int(v['pos']),
                    v['ref'],
                    v['alt'],
                    id=str(v.get('variant_id', ''))
                )
                variant_objects.append(variant_obj)
            else:
                # Assume it's already a kipoiseq.Variant
                variant_objects.append(v)

        # Sort variants by position
        variant_objects = sorted(variant_objects, key=lambda v: v.start)

        # Verify all variants are on the same chromosome
        chroms = set(v.chrom for v in variant_objects)
        if len(chroms) > 1:
            raise ValueError(f"All variants must be on same chromosome for local mode. Found: {chroms}")

        # Calculate the span of variants
        min_pos = min(v.start for v in variant_objects)
        max_pos = max(v.start for v in variant_objects)
        variant_span = max_pos - min_pos

        # Check if variants fit within sequence_length
        if variant_span > sequence_length:
            raise ValueError(
                f"Variant span ({variant_span}bp) exceeds sequence_length ({sequence_length}bp). "
                f"Use aggregated mode for distant variants."
            )

        # Create interval centered on the middle of the variant span
        center_pos = (min_pos + max_pos) // 2
        chrom = variant_objects[0].chrom
        interval = kipoiseq.Interval(chrom, center_pos, center_pos).resize(sequence_length)

        # Validate chromosome
        if self.fasta_extractor.is_valid_chromosome(interval.chrom) is None:
            return None, None

        # Extract sequences using kipoiseq's multi-variant support
        seq_extractor = kipoiseq.extractors.VariantSeqExtractor(reference_sequence=self.fasta_extractor)
        center = interval.center() - interval.start

        reference = seq_extractor.extract(interval, [], anchor=center)
        alternate = seq_extractor.extract(interval, variant_objects, anchor=center)

        return reference, alternate

    def can_use_local_mode(self, variants: list, sequence_length: int = 1024) -> bool:
        """
        Check if variants can be processed in local mode (single sequence).

        Local mode requires:
        - All variants on the same chromosome
        - Variant span < sequence_length

        Args:
            variants: List of variant dicts or kipoiseq.Variant objects
            sequence_length: Sequence length to use

        Returns:
            True if local mode can be used, False otherwise
        """
        if not variants or len(variants) == 0:
            return False

        if len(variants) == 1:
            return True

        # Extract chromosome and position info
        positions = []
        chroms = set()

        for v in variants:
            if isinstance(v, dict):
                chrom = v['chrom']
                if not chrom.startswith('chr'):
                    chrom = f"chr{chrom}"
                chroms.add(chrom)
                positions.append(int(v['pos']))
            else:
                chroms.add(v.chrom)
                positions.append(v.start)

        # Check all same chromosome
        if len(chroms) > 1:
            return False

        # Check span
        variant_span = max(positions) - min(positions)
        return variant_span < sequence_length

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
        if self.GTF is not None:
            return self.extract_sequence(variant, sequence_length=sequence_length), is_encoding(self.GTF, f"chr{chr}", pos, pos)
        return self.extract_sequence(variant, sequence_length=sequence_length)

class FastaStringExtractor:
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}
        synonynms_file = './root/data/chromAliases.txt'
        if not os.path.exists(synonynms_file):
            if os.path.exists('./root/data/chromAliases.txt'):
                synonynms_file = './root/data/chromAliases.txt'
            else:
                self.download_synonyms()
                synonynms_file = './root/data/chromAliases.txt'
        # mapping the NCBI chromosome names to UCSC chromosome names if necessary
        self.chrom_synonyms = self.read_synonyms(synonynms_file)

    def download_synonyms(self,url="http://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/chromAlias.txt.gz"):
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
