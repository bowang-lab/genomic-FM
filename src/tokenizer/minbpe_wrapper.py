from .minbpe import BasicTokenizer, RegexTokenizer
from tqdm import tqdm
from typing import List, Optional
from pathlib import Path


DEFAULT_VOCAB_SIZE = 256
COMMON_SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
# This file is used to import the tokenizer classes from the minbpe module
class MinBpeTokenizer:
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, special_tokens: Optional[List[str]] = None,
                 tokenizer_type: str = "Basic"):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else COMMON_SPECIAL_TOKENS
        self.tokenizer = BasicTokenizer() if tokenizer_type == "Basic" else RegexTokenizer()
        self.tokenizer_type="MinBPE"
        self.token_to_id = {}
        self.id_to_token = {}

    def load(self, file_path: Path):
        self.tokenizer.load(str(file_path))
        # Initialize the token-ID and ID-token mappings
        self._initialize_mappings()

    def train(self, sequences: List[str]):
        self.tokenizer.train_from_iterator(sequences, self.vocab_size)
        # Initialize the token-ID and ID-token mappings
        self._initialize_mappings()

    def _initialize_mappings(self):
        vocab = self.tokenizer.vocab
        self.token_to_id = vocab
        self.id_to_token = {id_: token for token, id_ in vocab.items()}

    def save(self, output_dir: Path, tokenizer_name: str):
        file_path = str((output_dir / f"{self.tokenizer_type}_{tokenizer_name}.json").resolve())
        self.tokenizer.save(file_path)

    def encode(self, sequence: str) -> List[int]:
        return self.tokenizer.encode(sequence)

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
            encoded = [self.tokenizer.encode(seq) for seq in batch]
            tokenized_sequences.extend([enc for enc in encoded])
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
