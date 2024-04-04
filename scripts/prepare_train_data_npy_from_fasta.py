import os
import numpy as np
from Bio import SeqIO


def tokenize_and_save_fasta_as_npy(fasta_file, seq_length, target_dir, chunk_size, tokenizer, padding_token=-1):
    """
    Tokenize sequences from a fasta file and save as .npy files.

    :param fasta_file: Path to the FASTA file.
    :param seq_length: The length to which sequences should be truncated or padded.
    :param target_dir: Directory where .npy files will be saved.
    :param chunk_size: The maximum size of each .npy file in number of sequences.
    :param tokenizer: An object with a tokenize method for tokenizing sequences.
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    buffer = []  # buffer to hold tokenized sequences
    file_count = 0

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        tokenized_seq = tokenizer.encode(sequence)

        # Break the sequence into multiple parts if longer than seq_length
        for i in range(0, len(tokenized_seq), seq_length):
            chunk = tokenized_seq[i:i + seq_length]
            if len(chunk) < seq_length:
                chunk += [padding_token] * (seq_length - len(chunk))
            buffer.append(chunk)

            if len(buffer) >= chunk_size:
                npy_file_path = os.path.join(target_dir, f"chunk_{file_count}.npy")
                np.save(npy_file_path, np.array(buffer))
                buffer = []  # Reset buffer
                file_count += 1

    # Save any remaining sequences in the buffer
    if buffer:
        npy_file_path = os.path.join(target_dir, f"chunk_{file_count}.npy")
        np.save(npy_file_path, np.array(buffer))

# Example usage
# tokenize_and_save_fasta_as_npy("genome.fasta", 1000, "npy_output", 10000, tokenizer)
