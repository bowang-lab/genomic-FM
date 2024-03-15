from src.datasets.ncbi_reference_genome.ncbi_dataset import add_tokens_to_fasta_from_gtf
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
    # Example usage (file path needs to be replaced with the actual file path)
    file_path = '/Users/lizehui/Desktop/workspace/genomic_vairants_benchmark/genomic-FM/root/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna'
    # chromosome_lengths = calculate_chromosome_lengths(file_path)
    # print(chromosome_lengths)
    human_fasta_file = 'root/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic.fna'
    human_gtf_file = 'root/data/GCF_000001405.40/genomic.gtf'
    extractor = parallel_process_gtf(human_fasta_file, human_gtf_file, "root/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic_modified.fna")
    # extractor.save_chrm_to_file("root/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic_modified_chr1.fna", "NC_000001.11")
    # extractor = add_tokens_to_fasta_from_gtf(human_fasta_file, human_gtf_file, "root/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic_modified.fna")
    # extractor.save_chrm_to_file("root/data/GCF_000001405.40/GCF_000001405.40_GRCh38.p14_genomic_modified_chr1.fna", "NC_000001.11")
