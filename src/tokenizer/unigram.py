from tokenizers import Tokenizer, models, trainers
from typing import Optional, List
from tqdm import tqdm
import argparse
import json
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer
from pathlib import Path
from collections import Counter

DEFAULT_VOCAB_SIZE = 32000  # Replace with your default vocab size
COMMON_SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]  # Replace with your common special tokens

class UnigramTokenizer:
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else COMMON_SPECIAL_TOKENS
        self.tokenizer = Tokenizer(models.Unigram())
        self.tokenizer_type = "Unigram"

    def train(self, sequences: List[str]):
        trainer = trainers.UnigramTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(sequences, trainer=trainer)

    def save(self, output_dir: Path, tokenizer_name: str):
        file_path = str((output_dir / f"{self.tokenizer_type}_{tokenizer_name}.json").resolve())
        self.tokenizer.save(file_path)

    def encode(self, sequence: str) -> List[int]:
        return self.tokenizer.encode(sequence).ids

    def encode_in_batches(self, sequences: List[str], batch_size: int = 1000) -> List[List[int]]:
        """
        Tokenizes the provided data using the given tokenizer.
    
        Parameters:
        - sequences: A list of strings (DNA sequences) to tokenize.
        - batch_size: Size of batches to process the data
    
        Returns:
        - List of tokenized sequences.
        """
        tokenized_sequences = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Tokenizing"):
            batch = sequences[i:i + batch_size]
            encoded = self.tokenizer.encode_batch(batch)
            tokenized_sequences.extend([enc.ids for enc in encoded])
        return tokenized_sequences

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

