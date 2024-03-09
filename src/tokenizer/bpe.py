import itertools
from pathlib import Path
from typing import Optional,List, Generator
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import sentencepiece as spm
import argparse
import glob
import random
from tqdm import tqdm
import tempfile
import random
from pathlib import Path

DEFAULT_VOCAB_SIZE = 32000
COMMON_SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

class BpeTokenizer:
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else COMMON_SPECIAL_TOKENS
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer_type="BPE"
        self.token_to_id = {}
        self.id_to_token = {}

    @classmethod
    def load(cls, file_path: Path):
        tokenizer_instance = cls()
        tokenizer_instance.tokenizer = Tokenizer.from_file(str(file_path))
        
        # Initialize the token-ID and ID-token mappings
        tokenizer_instance._initialize_mappings()
        
        return tokenizer_instance
        
    def train(self, sequences: List[str]):
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(sequences, trainer=trainer)
        
        # Initialize the token-ID and ID-token mappings
        self._initialize_mappings()

    def _initialize_mappings(self):
        vocab = self.tokenizer.get_vocab()
        self.token_to_id = vocab
        self.id_to_token = {id_: token for token, id_ in vocab.items()}
        
    def save(self, output_dir: Path, tokenizer_name: str):
        file_path = str((output_dir / f"{self.tokenizer_type}_{tokenizer_name}.json").resolve())
        self.tokenizer.save(file_path)

    def encode(self, sequence: str, add_special_tokens=True) -> List[int]:
        return self.tokenizer.encode(sequence, add_special_tokens=True).ids

    def encode_in_batches(self, sequences: List[str], batch_size: int = 1000) -> List[List[int]]:
        tokenized_sequences = []
        for i in tqdm(range(0, len(sequences), batch_size), desc="Tokenizing"):
            batch = sequences[i:i + batch_size]
            encoded = self.tokenizer.encode_batch(batch)
            tokenized_sequences.extend([enc.tokens for enc in encoded])
        return tokenized_sequences

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def convert_tokens_to_ids(self, tokens) -> List[int]:
        unk_id = self.token_to_id.get("[UNK]", None)
        if isinstance(tokens[0], list):
            return [[self.token_to_id.get(token, unk_id) for token in tokens] for tokens in tokens]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def convert_ids_to_tokens(self, ids_batch: List[List[str]]) -> List[List[int]]:
        """
        Converts a batch of sequences of IDs into a batch of sequences of tokens.

        Parameters:
        - ids_batch (List[List[str]]): A batch of sequences, where each sequence is a list of token IDs.

        Returns:
        - List[List[int]]: A batch of sequences, where each sequence is a list of tokens.
        """
        return [[self.id_to_token[id_] for id_ in ids] for ids in ids_batch]


def main():
    parser = argparse.ArgumentParser(description="Command line interface for tokenizer creation")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Name of the tokenizer")
    parser.add_argument("--tokenizer-type", type=str, required=True, help="Type of tokenizer")
    parser.add_argument("--input-dir", type=str, required=True, help="Input directory")
    parser.add_argument("--evaluate-dir", type=str, required=True, help="Evaluation directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Vocab size to be included")
    parser.add_argument("--samples-per-file", type=int, default=100, help="Number of sequences to sample from each file")
    parser.add_argument("--limit-files", type=int, default=10, help="Number of files containing sequences to limit tokenizer training")
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

    bpe_tokenizer = BpeTokenizer(vocab_size=args.vocab_size)
    bpe_tokenizer.train(sequences)
    bpe_tokenizer.save(output_dir, args.tokenizer_name)

    # Evaluate the tokenizer
    print("Evaluate tokenizer...")
    evaluation_data = load_sequences(evaluate_dir,limit_files=10,samples_per_file=1000)
    tokenized_data = bpe_tokenizer.encode_in_batches(evaluation_data)
    token_statistics = calculate_token_statistics(tokenized_data)
    
    with open(output_file_path, 'w') as file:
        json.dump(token_statistics, file, indent=4)
    
    plot_and_save_evaluation_results(token_statistics, output_dir)


if __name__ == "__main__":
    main()
