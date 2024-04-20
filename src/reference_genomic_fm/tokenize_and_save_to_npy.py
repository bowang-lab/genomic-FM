import os
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

MAX_sentence_length = 5000000

def remove_prepending_n(dna_sequence):
    return dna_sequence.lstrip('N').rstrip('N')

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

    for record in tqdm(SeqIO.parse(fasta_file, "fasta")):
        sequence = str(record.seq)
        sequence = remove_prepending_n(sequence)
        tokenized_seq = []
        print(f"ID {record.id} - Original length: {len(sequence)}")
        # break the sequence into multiple parts if longer than MAX_sentence_length
        if len(sequence) > MAX_sentence_length:
            for i in range(0, len(sequence), MAX_sentence_length):
                original_chunk = sequence[i:i + MAX_sentence_length]
                tokenized_chunk = tokenizer.encode(original_chunk)
                print(f"Tokenized length: {len(tokenized_chunk)}")
                tokenized_seq.append(tokenized_chunk)
        else:
            tokenized_seq.append(tokenizer.encode(sequence))
        # Break the sequence into multiple parts if longer than seq_length
        for ele in tokenized_seq:
            for i in range(0, len(ele), seq_length):
                chunk = ele[i:i + seq_length]
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

def save_numpy_files_in_binary(source_directory, target_directory, dtype_y=np.uint16):
    """
    Reads all .npy files in the specified source directory, converts them to a specific dtype,
    and saves them in binary format in the target directory.

    Args:
    source_directory (str): The path to the directory containing .npy files.
    target_directory (str): The path to the directory where binary files will be saved.
    dtype_y (data-type): The data type to which numpy arrays should be converted before saving.
    """
    # Create the target directory if it does not exist
    os.makedirs(target_directory, exist_ok=True)

    # Check each file in the source directory
    for filename in tqdm(os.listdir(source_directory)):
        if filename.endswith(".npy"):
            # Full path to the current file
            file_path = os.path.join(source_directory, filename)

            # Load the numpy array from .npy file
            data = np.load(file_path)

            # replace all -1 with 78 (padding token)
            data[data == -1] = 78

            # Convert data to specified data type
            data = np.array(data, dtype=dtype_y)

            # Prepare the binary data
            binary_data = data.tobytes()

            # New filename for the binary file
            binary_file_path = os.path.join(target_directory, os.path.splitext(filename)[0] + '.npy')

            # Write the binary data to a new file in the target directory
            with open(binary_file_path, 'wb') as f:
                f.write(binary_data)

            print(f"Saved {binary_file_path}")
