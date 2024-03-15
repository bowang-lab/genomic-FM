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

