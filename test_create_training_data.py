from src.reference_genomic_fm.preprocess_inserted_fasta import preprocess_and_save_sequences
from src.reference_genomic_fm.train_minbpe_tokenizer import train_minbpe_tokenizer
from src.reference_genomic_fm.tokenize_and_save_to_npy import tokenize_and_save_fasta_as_npy
from src.tokenizer.utils import load_sequences
from src.tokenizer.minbpe_wrapper import MinBpeTokenizer
from pathlib import Path


# preprocess_and_save_sequences(['/mnt/data/shared/icml_prepare_dataset/annotated_fasta_train_tokenizer'], '/home/zl6222/repositories/genomic-FM/root/data/human_chrom_train_tokenizer/train_tokenizer.fna')
# input_dir = Path('/home/zl6222/repositories/genomic-FM/root/data/human_chrom_train_tokenizer/')
# sequences = load_sequences(
#             input_dir,
#             limit_files=None,
#             random_files=False
#         )
# tokenizer = train_minbpe_tokenizer(sequences, 4096)

## load tokenizer
tokenizer = MinBpeTokenizer(vocab_size=4096)
tokenizer.load('/home/zl6222/repositories/genomic-FM/root/data/tokenizer/MinBPE_minbpe_torch.json.model')
tokenize_and_save_fasta_as_npy('./root/data/human_chrom/human_chromosomes.fna', 2048, '/mnt/data/shared/icml_prepare_dataset/npy_data', 1000, tokenizer)
