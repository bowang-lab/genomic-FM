from src.datasets.ncbi_reference_genome.ncbi_dataset_parrallel import parallel_process_gtf


def calculate_chromosome_lengths(file_path):
    with open(file_path, 'r') as file:
        lengths = {}
        current_chromosome = None
        sequence_length = 0

        for line in file:
            if line.startswith('>'):  # Header line, indicating a new chromosome
                if current_chromosome is not None:
                    # Save length of the previous chromosome
                    lengths[current_chromosome] = sequence_length
                # Update current chromosome name and reset length
                current_chromosome = line[1:].strip()  # Remove '>' and any trailing newline/whitespace
                sequence_length = 0
            else:
                sequence_length += len(line.strip())  # Add length of the line, excluding newline/whitespace

        # Save the last chromosome's length
        if current_chromosome is not None:
            lengths[current_chromosome] = sequence_length

    return lengths

if __name__ == '__main__':
    human_fasta_file = 'root/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna'
    human_gtf_file = 'root/data/GCF_000001405.40/genomic.gtf'
    extractor = parallel_process_gtf(human_fasta_file, human_gtf_file, "root/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic_modified.fna")
