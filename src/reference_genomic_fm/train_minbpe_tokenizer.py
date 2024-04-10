
from ..tokenizer.minbpe_wrapper import MinBpeTokenizer
from pathlib import Path
from ..tokenizer.bpe import BpeTokenizer


def train_minbpe_tokenizer(sequences, vocab_size=4096, output_dir='./root/data/tokenizer/', tokenizer_name='minbpe_torch'):
    if tokenizer_name=='minbpe_torch':
        tokenizer = MinBpeTokenizer(vocab_size=vocab_size)
    else:
        tokenizer = BpeTokenizer(vocab_size=vocab_size)
    tokenizer.train(sequences)
    output_dir = Path(output_dir)
    tokenizer.save(output_dir, tokenizer_name)
    print("Tokenizer trained and saved")
    # Evaluate tokenizer...
    print(f"Tokenizer vocab size: {tokenizer.token_to_id}")
    return tokenizer
