import pysam
from pathlib import Path
import argparse
import multiprocessing
import pandas as pd
import os
from Bio import SeqIO
import json
from typing import Optional, List, Tuple, Generator
import random
import vcf
from tqdm import tqdm
import shutil

class Sequences:
    # Class to handle storage and formatting of text data
    def __init__(self, sentences: Optional[List[str]] = None, metadata: Optional[dict] = None, use_special_tokens: bool = False, max_length: Optional[int] = None) -> None:
        self.sentences = sentences if sentences is not None else []
        self.metadata = metadata if metadata is not None else {}
        self.use_special_tokens = use_special_tokens
        self.max_length = max_length

    def replace_n_with_unk(self, sentence: str) -> str:
        return sentence.replace('N', '[UNK]')

    def remove_stop_codons(self, sentence: str) -> str:
        return sentence.replace('*', '')

    def apply_special_tokens(self, sentence: str) -> str:
        modified_sentence = f"[CLS]{sentence}[SEP]"
        return modified_sentence

    def pad_sequence(self, sentence: str) -> str:
        # Pad the sequence to the max_length
        padding_length = self.max_length - len(sentence)
        if padding_length > 0:
            sentence += "[PAD]" * padding_length
        return sentence
        
    def to_list(self) -> List[str]:
        # Return the list of individual sequences
        return self.sentences
        
    def append(self, sentence: str) -> None:
        modified_sentence = self.remove_stop_codons(sentence)
        modified_sentence = self.replace_n_with_unk(sentence)
        if self.use_special_tokens:
            modified_sentence = self.apply_special_tokens(modified_sentence)
            if self.max_length is not None:
                modified_sentence = self.pad_sequence(modified_sentence)
        self.sentences.append(modified_sentence)

    def to_text(self) -> str:
        # Convert all sentences to a single text string
        return "\n".join(self.sentences)

    def to_jsonl(self) -> str:
        # Convert the sequence to a JSON line format
        return json.dumps({"text": self.sentences, **self.metadata})

    def save_to_file(self, file_path: str, output_format: str) -> None:
        # Save the sequences to a file in the specified format (txt or jsonl)
        with open(file_path, 'w') as f:
            if output_format == 'txt':
                f.write(self.to_text())
            elif output_format == 'jsonl':
                f.write(self.to_jsonl())
            else:
                raise ValueError("Invalid output format. Choose 'txt' or 'jsonl'.")

def process_contig(reference_fasta, vcf_file, output_fasta, contig):
    # Process a single contig from a reference FASTA file and VCF file, saving the modified sequence to a new file
    output_file = output_fasta / f'{contig}.fasta'

    if output_file.exists():
        print(f"Output file for contig {contig} already exists. Skipping.")
        return

    try:
        # Open FASTA and VCF files for reading data
        fasta = pysam.FastaFile(reference_fasta)
        variants = pysam.VariantFile(vcf_file)

        # Fetch the sequence for the given contig and apply variants
        sequence = list(fasta.fetch(contig))
        for record in variants.fetch(contig):
            if not record.alts or record.alts[0] == '<*>':
                continue

            for alt in record.alts:
                if len(record.ref) == len(alt):
                    sequence[record.pos - 1] = alt  # Adjusted for 1-based indexing

        # Write the modified sequence to the output file
        with open(output_file, 'w') as output:
            output.write(f'>{contig}\n')
            output.write('\n'.join(sequence))

    except ValueError as e:
        print(f"Skipping contig {contig} as it's not found: {e}")
    finally:
        # Close the FASTA and VCF files
        fasta.close()
        variants.close()

def extract_random_sequences(
    fasta_file: Path,
    output_file: Path,
    output_format: str = 'txt',
    desired_length: int = 1000,
    use_special_tokens: bool = True,
    sentences_bounds: Tuple[int, int] = (10, 20),
    lengths_bounds: Tuple[int, int] = (1000, 10000),
):

    assert isinstance(sentences_bounds, tuple) and len(sentences_bounds) == 2, "sentences_bounds must be a tuple with two integers"
    assert isinstance(lengths_bounds, tuple) and len(lengths_bounds) == 2, "lengths_bounds must be a tuple with two integers"

    
    # Extracts sequences from a FASTA file and saves them to an output file in the specified format
    sequence_record = next(SeqIO.parse(fasta_file, "fasta"))  # Reading the first sequence record from the FASTA file
    chr_sequence = sequence_record.seq  # Storing the chromosome sequence

    # Chromosome length
    C = len(chr_sequence)

    # Create a Sequences object to store sequences
    sequences = Sequences(use_special_tokens=use_special_tokens, max_length=desired_length)

    # Generate sequences with random segments of the sequence
    for _ in range(5):  # Looping a fixed number of times (10) to generate sequences
        q = random.randint(0, 5000)  # random start position within a reasonable range
        while q < C:
            s = random.randint(*sentences_bounds)  # number of sequences
            for _ in range(s):
                l = random.randint(*lengths_bounds)  # length of each sentence
                if q + l > C:
                    l = C - q  # Adjust length if it exceeds the chromosome length
                sequences.append(str(chr_sequence[q: q + l]).upper())  # Append the sequence segment to the sequences
                q += l
                if q >= C:  # end of sequence
                    break

    # Save the sequences to the output file using the specified format
    sequences.save_to_file(output_file, output_format)

def extract_variant_sequences(
    vcf_file, 
    fasta_file, 
    left_padding, 
    right_padding, 
    output_file,     
    output_format: str = 'txt',
    use_special_tokens: bool = True,
):
    # Extracts sequences from a FASTA file based on VCF records with specified padding, then saves to an output file
    # Using a context manager for reading FASTA file
    with open(fasta_file, "r") as fasta_handle:
        fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_handle, "fasta"))

    # Using a context manager for the VCF file
    with open(vcf_file, 'r') as vcf_handle:
        vcf_reader = vcf.Reader(vcf_handle)

        # Create a Sequences object to store sequences
        sequences = Sequences(use_special_tokens=use_special_tokens)

        # Iterate over each record in the VCF file
        for record in vcf_reader:
            chrom = record.CHROM
            pos = record.POS
            
            if chrom in fasta_sequences:
                # Extract the sequence for the chromosome
                chrom_sequence = fasta_sequences[chrom].seq

                # Calculate start and end positions for slicing, considering padding
                # Adjusting for 0-based indexing in FASTA
                start = max(0, pos - 1 - left_padding)
                end = pos - 1 + right_padding
                # Extract the sequence with padding
                seq_with_padding = chrom_sequence[start:end]
                sequences.append(str(seq_with_padding))

        # Save the sequences to the output file
        sequences.save_to_file(output_file, output_format)

def index_vcf(vcf_filename):
    """
    Index a VCF file using pysam.tabix_index, keeping the original file.

    Parameters:
    vcf_filename (str): Path to the VCF file.
    """
    # Copy the original VCF to a new file with .gz extension
    tmp_vcf_filename = vcf_filename + ".tmp"

    shutil.copyfile(vcf_filename, tmp_vcf_filename)

    # Index the gzipped VCF file
    pysam.tabix_index(vcf_filename, preset="vcf", force=True)

    shutil.move(vcf_filename+".tmp", vcf_filename)
    
# Function to convert VCF variants to a FASTA format
def convert_vcf_to_fasta(reference_fasta, vcf_file, output_dir):
    # Open the reference FASTA file and get the list of contigs
    fasta = pysam.FastaFile(reference_fasta)
    contigs = fasta.references
    fasta.close()

    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each contig in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        pool.starmap(process_contig, [(reference_fasta, vcf_file, output_dir, contig) for contig in contigs])

def main():
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Genomic Sequence Processing Tool')
    
    # Define arguments for the script
    parser.add_argument('--reference_fasta', type=str, required=True, help='Path to the reference FASTA file.')
    parser.add_argument('--vcf_file', type=str, required=False, help='Path to the VCF file.')
    parser.add_argument('--vcf_to_fasta', default=False, help='Indicator for converting vcf to fasta.')
    parser.add_argument('--fasta_dir', type=str, required=True, help='Output directory for the processed files.')
    parser.add_argument('--sequence_dir', type=str, required=True, help='Output directory for the processed sequences.')
    parser.add_argument('--left_padding', type=int, default=5000, help='Left padding for sequences.')
    parser.add_argument('--right_padding', type=int, default=5000, help='Right padding for sequences.')
    parser.add_argument('--output_format', type=str, default='txt', choices=['txt', 'json'], help='Output format for sequences.')
    parser.add_argument('--process_type', type=str, required=True, choices=['variant', 'random'], help='Type of processing: "variant" or "random".')
    parser.add_argument('--max_desired_length', type=int, default=10000, help='Maximum desired length.')
    parser.add_argument('--use_special_tokens', default=False, help='Use special tokens.')

    # Parse command-line arguments
    args = parser.parse_args()
   
    # Process according to the specified type
    if args.vcf_to_fasta:
        print("[ Step 1 ] Index vcf")
        # Index the VCF file before processing
        index_vcf(args.vcf_file)
        indexed_vcf_file = args.vcf_file + ".gz"
        # Convert VCF variants to FASTA format
        print("[ Step 2 ] Convert vcf to fasta")
        convert_vcf_to_fasta(args.reference_fasta, indexed_vcf_file, args.fasta_dir)
  
    # Set up directories for processing
    fasta_dir = Path(args.fasta_dir)
    sequence_dir = Path(args.sequence_dir)
    
    print("[ Step 3 ] Convert fastas to sequences")
    # Process each FASTA file in the parent directory
    for fasta_file in fasta_dir.glob('*.fasta'):
        if fasta_file.stem.startswith("NC") or fasta_file.stem.startswith("GL") or fasta_file.stem.startswith("hs"):
            continue  # Skip variables with prefixes 'GL' or 'hs'
        print("Chromosome " + fasta_file.stem)
        output_file = sequence_dir / f"{fasta_file.stem}_sequences.{args.output_format}"
            
        if args.process_type == 'variant':
            extract_variant_sequences(args.vcf_file, fasta_file, args.left_padding, args.right_padding, output_file, args.output_format, args.use_special_tokens)
        elif args.process_type == 'random':
            extract_random_sequences(fasta_file, output_file, args.output_format, args.max_desired_length, args.use_special_tokens)
        else:
            raise ValueError("Invalid process type. Choose 'variant' or 'random'.")

if __name__ == '__main__':
    main()
