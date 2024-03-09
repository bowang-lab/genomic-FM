import os
from typing import List, Optional
from pathlib import Path
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import collections
import seaborn as sns

def load_sequences(
    input_dir: Optional[Path] = None, 
    cache_dir: Optional[Path] = None,
    hf_dataset: Optional[str] = None, 
    hf_dataset_config: Optional[str] = None, 
    hf_dataset_split: str = "train", 
    pattern: str = "*/*sequences.txt", 
    samples_per_file: Optional[int] = None, 
    random_files: bool = False,
    limit_files: Optional[int] = None,
) -> List[str]:
    """
    Loads sequences from text files located in the specified input directory.

    Parameters:
    - input_dir (Path): Directory containing text files with sequences.
    - pattern (str): Pattern to match the files. Default is "*sequences.txt".
    - limit_files (Optional[int]): Maximum number of files to read. If None, all files are read.

    Returns:
    - List of strings: Loaded sequences.
    """
    sequences = []

    if hf_dataset:
        # Load sequences from a Hugging Face dataset
        dataset = load_dataset(hf_dataset, hf_dataset_config, split=hf_dataset_split,cache_dir=cache_dir)
        sequences = [str(data["sequence"]) for data in dataset]
    elif input_dir:
        files = list(input_dir.glob(pattern))
    
        # Randomly select files if random_files is True
        if random_files and len(files) > limit_files:
            files = random.sample(files, limit_files)
    
        file_count = 0
        for file_path in files:
            if limit_files is not None and file_count >= limit_files:
                break
    
            with open(file_path, 'r') as file:
                file_sequences = [line.strip() for line in file if line.strip()]
                # Randomly sample sequences if samples_per_file is specified
                if samples_per_file and len(file_sequences) > samples_per_file:
                    file_sequences = random.sample(file_sequences, samples_per_file)
                sequences.extend(file_sequences)
    
            file_count += 1
    return sequences

def plot_and_save_evaluation_results(evaluation_results, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    # Plot total_tokens and unique_tokens
    total_tokens = [results['total_tokens'] for vocab_size, results in evaluation_results.items()]
    unique_tokens = [results['unique_tokens'] for vocab_size, results in evaluation_results.items()]
    vocab_sizes = list(evaluation_results.keys())

    plt.figure(figsize=(8, 4))  # Smaller figure size
    plt.subplot(1, 2, 1)
    sns.barplot(x=vocab_sizes, y=total_tokens)
    plt.title('Total Tokens vs Vocab Size')
    plt.xlabel('Vocab Size')
    plt.ylabel('Total Tokens')

    plt.subplot(1, 2, 2)
    sns.barplot(x=vocab_sizes, y=unique_tokens)
    plt.title('Unique Tokens vs Vocab Size')
    plt.xlabel('Vocab Size')
    plt.ylabel('Unique Tokens')

    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_total_unique_tokens_plot.png", dpi=300)
    plt.close()

    # Plot token length distribution for each vocabulary size
    for vocab_size, results in evaluation_results.items():
        token_length_dist = results['token_length_distribution']
        # Sort the token lengths
        sorted_lengths = sorted(token_length_dist.keys(), key=lambda x: int(x))
        frequencies = [token_length_dist[length] for length in sorted_lengths]
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=sorted_lengths, y=frequencies)
        plt.title(f'Token Length Distribution for Vocab Size: {vocab_size}')
        plt.xlabel('Token Length')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / f"{name}_token_length_distribution_vocab_{vocab_size}.png", dpi=300)
        plt.close()

    # Plot token distribution for each vocabulary size
    for vocab_size, results in evaluation_results.items():
        token_dist = results['token_distribution']
        tokens, frequencies = zip(*sorted(token_dist.items(), key=lambda item: item[1], reverse=True)[:20])  # Top 20 tokens

        plt.figure(figsize=(12, 5))  # Adjust figure size as needed
        sns.barplot(x=list(tokens), y=list(frequencies))
        plt.title(f'Top Token Distribution for Vocab Size: {vocab_size}')
        plt.xlabel('Token')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.savefig(output_dir / f"{name}_token_distribution_vocab_{vocab_size}.png", dpi=300)
        plt.close()

def calculate_token_statistics(tokenized_data):
    """
    Calculates token statistics from the tokenized data.

    Parameters:
    - tokenized_data: List of lists, where each inner list is a sequence of tokens.

    Returns:
    - Dictionary with token statistics.
    """
    token_freq = collections.Counter()
    token_lengths = collections.Counter()
    total_tokens = 0
    nucleotide_counts = {'A': 0, 'T': 0, 'C': 0, 'G': 0}

    for sequence in tokenized_data:
        for token in sequence:
            token_freq[token] += 1
            token_lengths[len(token)] += 1
            total_tokens += 1
            nucleotide_counts['A'] += token.count('A')
            nucleotide_counts['T'] += token.count('T')
            nucleotide_counts['C'] += token.count('C')
            nucleotide_counts['G'] += token.count('G')

    # Calculating token distribution and other stats
    token_dist = {token: count / total_tokens for token, count in token_freq.items()}
    avg_token_length = np.mean(list(token_lengths.keys())) if token_lengths else 0
    std_dev_token_length = np.std(list(token_lengths.keys())) if token_lengths else 0
    token_length_distribution = {length: count for length, count in token_lengths.items()}

    statistics = {
        "total_tokens": total_tokens,
        "unique_tokens": len(token_freq),
        "most_common_token": token_freq.most_common(1)[0] if token_freq else ('', 0),
        "token_distribution": token_dist,
        "average_token_length": avg_token_length,
        "std_dev_token_length": std_dev_token_length,
        "nucleotide_counts": nucleotide_counts,
        "token_length_distribution": token_length_distribution
    }

    return statistics

