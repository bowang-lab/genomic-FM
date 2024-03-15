from src.tokenizer.utils import load_sequences
from src.tokenizer.minbpe_wrapper import MinBpeTokenizer
from pathlib import Path


input_dir = Path('./root/data/train_fasta/')
sequences = load_sequences(
        input_dir,
        limit_files=1,
        random_files=True
    )

tokenizer = MinBpeTokenizer(vocab_size=256)
tokenizer.train(sequences)
output_dir = Path('./root/data/tokenizer/')
tokenizer.save(output_dir, 'minbpe')
print("Tokenizer trained and saved")
# Evaluate tokenizer...
print(f"Tokenizer vocab size: {tokenizer.token_to_id}")
from scripts.prepare_train_data_npy_from_fasta import tokenize_and_save_fasta_as_npy


fasta_file = './root/data/train_fasta/NW_003315958.1.fasta'
seq_length = 100
target_dir = './root/data/npy_output/'
chunk_size = 10000
tokenize_and_save_fasta_as_npy(fasta_file, seq_length,
                               target_dir, chunk_size, tokenizer)

# test load npy
import numpy as np

# Replace 'path_to_file.npy' with the actual file path
data = np.load('root/data/npy_output/chunk_0.npy')

print(data)

# Use OLMo to train a genomic language model
# https://github.com/allenai/OLMo/blob/0b6e28c0c97d9d3b97c41427f151951a1c44048e/configs/official/OLMo-1B.yaml
