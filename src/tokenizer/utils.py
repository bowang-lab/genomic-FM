import os
from typing import List, Optional
from pathlib import Path
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import collections
import seaborn as sns
from typing import Generator, Optional
import random

def load_sequences(
    input_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    hf_dataset: Optional[str] = None,
    hf_dataset_config: Optional[str] = None,
    hf_dataset_split: str = "train",
    pattern: str = "*.fna",
    samples_per_file: Optional[int] = None,
    random_files: bool = False,
    limit_files: Optional[int] = None,
) -> Generator[str, None, None]:
    """
    Optimized generator function to load sequences from FASTA files or Hugging Face datasets.
    Yields sequences as they are loaded, improving memory efficiency for large datasets.
    """
    
    if hf_dataset:
        dataset = load_dataset(hf_dataset, hf_dataset_config, split=hf_dataset_split, cache_dir=cache_dir)
        for data in dataset:
            yield str(data["sequence"])
    elif input_dir:
        all_files = input_dir.glob('**/' + pattern)
        files = list(all_files) if limit_files is not None or random_files else all_files

        if random_files and limit_files is not None:
            files = random.sample(files, min(limit_files, len(files)))
        elif limit_files is not None:
            files = files[:limit_files]

        for file_path in files:
            with open(file_path, 'r') as file:
                current_sequence = ""
                for line in file:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_sequence:  # Yield the previous sequence if exists
                            yield current_sequence
                            current_sequence = ""  # Reset for a new sequence
                    else:
                        current_sequence += line
                if current_sequence:  # Yield the last sequence in the file
                    yield current_sequence


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

