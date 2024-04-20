from src.reference_genomic_fm.preprocess_inserted_fasta import preprocess_and_save_sequences
from src.reference_genomic_fm.train_minbpe_tokenizer import train_minbpe_tokenizer
from src.reference_genomic_fm.tokenize_and_save_to_npy import tokenize_and_save_fasta_as_npy, save_numpy_files_in_binary
from src.tokenizer.utils import load_sequences
from src.tokenizer.minbpe_wrapper import MinBpeTokenizer
from pathlib import Path


# preprocess_and_save_sequences(['/mnt/data/shared/icml_prepare_dataset/annotated_fasta'], '/mnt/data/shared/icml_prepare_dataset/human_chrom/human.fna')
# input_dir = Path('/mnt/data/shared/icml_prepare_dataset/human_chrom/human.fna')
# sequences = load_sequences(
#             input_dir,
#             limit_files=None,
#             random_files=False
#         )
# tokenizer = train_minbpe_tokenizer(sequences, 4096)

## load tokenizer
# tokenizer = MinBpeTokenizer(vocab_size=4096)
# tokenizer.load('/home/zl6222/repositories/genomic-FM/root/data/tokenizer/MinBPE_minbpe_torch.json.model')
# tokenize_and_save_fasta_as_npy('/mnt/data/shared/icml_prepare_dataset/human_chrom/human.fna', 2048, '/mnt/data/shared/icml_prepare_dataset/npy_data', 5000, tokenizer)

## save numpy files in binary
save_numpy_files_in_binary('/mnt/data/shared/icml_prepare_dataset/npy_data', '/mnt/data/zl6222/genomic-fm/human')
