import argparse
import os
from collections import Counter
from typing import Optional, List, Generator
from pathlib import Path
import json
from itertools import chain

class KmerTokenizer:
    def __init__(self, k: int, overlap: bool = True):
        self.k = k
        self.step = 1 if overlap else k
        self.vocab = {}
        self.inv_vocab = {}
        self.tokenizer_type = "Kmer"

    def build_vocab(self, sequences: List[str]) -> None:
        kmer_counts = Counter()

        for seq in sequences:
            for i in range(0, len(seq) - self.k + 1, self.step):
                kmer = seq[i:i + self.k]
                kmer_counts[kmer] += 1

        self.vocab = {kmer: idx for idx, (kmer, _) in enumerate(kmer_counts.items())}
        self.inv_vocab = {idx: kmer for kmer, idx in self.vocab.items()}

    def save(self, output_dir: Path, tokenizer_name: str):
        file_path = str((output_dir / f"{self.tokenizer_type}_{tokenizer_name}.json").resolve())
        with open(file_path, 'w') as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=4)

    def encode(self, sequence: str) -> List[str]:  # Returning List[str] instead of List[int] for k-mers
        return [sequence[i:i + self.k] for i in range(0, len(sequence) - self.k + 1, self.step)]

    def encode_in_batches(self, sequences: List[str], batch_size: int = 100) -> Generator[List[str], None, None]:
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            encoded_batch = [self.encode(sequence) for sequence in batch]
            # Flatten the list of lists into a single list
            flattened_encoded_batch = list(chain.from_iterable(encoded_batch))
            yield flattened_encoded_batch

    def decode(self, token_ids: List[int]) -> str:
        return ''.join(self.inv_vocab.get(idx, '') for idx in token_ids)

def main():
    parser = argparse.ArgumentParser(description="K-mer Tokenizer for DNA Sequences")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing DNA sequence files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--evaluate-dir", type=str, required=True, help="Evaluation directory")
    parser.add_argument("--k", type=int, default=3, help="K-mer length")
    parser.add_argument("--overlap", action='store_true', help="Generate overlapping k-mers (default)")
    parser.add_argument("--no-overlap", dest='overlap', action='store_false', help="Generate non-overlapping k-mers")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Name of the tokenizer")
    parser.add_argument("--limit-files", type=int, default=10, help="Number of files containing sequences to limit tokenizer training")

    parser.set_defaults(overlap=True)

    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    evaluate_dir = Path(args.evaluate_dir)

    sequences = load_sequences(
        input_dir, 
        limit_files=args.limit_files, 
        samples_per_file=args.samples_per_file, 
        random_files=True
    )

    kmer_tokenizer = KmerTokenizer(args.k, args.overlap)
    kmer_tokenizer.build_vocab(sequences)
    kmer_tokenizer.save(output_dir, args.tokenizer_name)
    
    # Evaluate the tokenizer
    print("Evaluate tokenizer...")
    evaluation_data = load_sequences(evaluate_dir,limit_files=10,samples_per_file=1000)
    tokenized_data = kmer_tokenizer.encode_in_batches(evaluation_data)
    token_statistics = calculate_token_statistics(tokenized_data)
    
    with open(output_file_path, 'w') as file:
        json.dump(token_statistics, file, indent=4)
    
    plot_and_save_evaluation_results(token_statistics, output_dir)

if __name__ == "__main__":
    main()
