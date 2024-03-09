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

def main():
    parser = argparse.ArgumentParser(description="Command line interface for tokenizer creation")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="Name of the tokenizer")
    parser.add_argument("--tokenizer-type", type=str, required=True, choices=['BPE', 'Unigram'], help="Type of tokenizer (BPE or Unigram)")
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

    tokenizer = UnigramTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(sequences)
    tokenizer.save(output_dir, args.tokenizer_name)

    # Evaluate the tokenizer
    print("Evaluate tokenizer...")
    evaluation_data = load_sequences(evaluate_dir,limit_files=10,samples_per_file=1000)
    tokenized_data = tokenizer.encode_in_batches(evaluation_data)
    token_statistics = calculate_token_statistics(tokenized_data)
    
    output_file_path = output_dir / f"{args.tokenizer_type}_{args.tokenizer_name}_statistics.json"
    with open(output_file_path, 'w') as file:
        json.dump(token_statistics, file, indent=4)
    
    plot_and_save_evaluation_results(token_statistics, output_dir)

if __name__ == "__main__":
    main()

