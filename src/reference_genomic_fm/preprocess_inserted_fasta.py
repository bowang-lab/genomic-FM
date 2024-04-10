from ..tokenizer.utils import load_sequences
from pathlib import Path
from tqdm import tqdm


def save_to_chromosome_file(sequences, output_file):
    chrom_index = 0
    with open(output_file, 'w') as out_f:
        for sequence in sequences:
            out_f.write(f'>{chrom_index}\n')
            out_f.write(''.join(sequence) + '\n')
            chrom_index += 1
    print(f"Saved to {output_file}")

def preprocess_sequences(path_to_fasta, limit_files=None, random_files=False):
    input_dir = Path(path_to_fasta)
    sequences = load_sequences(
            input_dir,
            limit_files=limit_files,
            random_files=random_files
        )
    standard = ['A','T','C','G', 'N', 'a', 't', 'c', 'g', 'n']
    new_sequences = []
    # convert a,t,c,g,n to A,T,C,G,N
    for seq in tqdm(sequences):
        new_seq = []
        for i in range(len(seq)):
            if seq[i] in standard:
                new_seq.append(seq[i].upper())
            else:
                new_seq.append(seq[i])
        new_sequences.append(new_seq)
    return new_sequences

def preprocess_and_save_sequences(path_to_fasta, output_file, limit_files=None, random_files=False):
    all_sequences = []
    for file in path_to_fasta:
        sequences = preprocess_sequences(file, limit_files, random_files)
        all_sequences.extend(sequences)
    save_to_chromosome_file(all_sequences, output_file)
