# from tqdm.contrib.concurrent import process_map  # Use this instead of Pool
import multiprocessing
import logging
from contextlib import closing
import tqdm
from .ncbi_dataset import NCBIFastaStringExtractor
import os


def split_gtf_file(gtf_file, output_file):
    chromosomes_data = {}
    with open(gtf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            chromosome = line.split('\t')[0]
            if chromosome not in chromosomes_data:
                chromosomes_data[chromosome] = []
            chromosomes_data[chromosome].append(line)
    # take only the top shortest 5 chromosomes
    chromosomes_data = dict(sorted(chromosomes_data.items(), key=lambda item: len(item[1])))
    # if in the folder of the output file, there is chromosome fasta, remove the chromosome from the dictionary
    for chrom in list(chromosomes_data.keys()):
        chrome_file = output_file + f'_{chrom}.fna'
        if os.path.exists(chrome_file):
            chromosomes_data.pop(chrom)
    return chromosomes_data

def save_chrm_to_file(sequence, output_file, chrm):
    output_file_chrom = output_file + f'_{chrm}.fna'
    with open(output_file_chrom, 'w') as out_f:
        out_f.write(f'>{chrm}\n')
        out_f.write(''.join(sequence) + '\n')  # Join the list into a string
    print(f"Saved to {output_file}")

def process_chromosome_data(gtf_lines, extractor):
    tokens_to_insert = []

    def extract_biotype(attributes_str):
        biotype_key = 'transcript_biotype "'
        start_index = attributes_str.find(biotype_key)
        if start_index == -1:
            return None  # Return None if biotype not found

        start_index += len(biotype_key)
        end_index = attributes_str.find('"', start_index)
        return attributes_str[start_index:end_index] if end_index != -1 else None

    def insert_tokens_and_reset():
        for chrom, position, token in sorted(tokens_to_insert):
            extractor.insert_token(chromosome=chrom, position=position, token=token)
        tokens_to_insert.clear()

    for line in tqdm.tqdm(gtf_lines):
        if line.startswith('#'):
            continue

        parts = line.strip().split('\t')
        chrom, feature_type, start, end, attributes = parts[0], parts[2], int(parts[3]), int(parts[4]), parts[8]

        if feature_type == 'gene':
            if tokens_to_insert:
                insert_tokens_and_reset()
            tokens_to_insert.append((chrom, start, '<gene_start>'))
            tokens_to_insert.append((chrom, end + 1, '<gene_end>'))
        elif feature_type in ['exon', 'CDS', 'start_codon', 'stop_codon']:
            if feature_type == 'exon':
                biotype = extract_biotype(attributes)
                token_start = f'<{biotype}_{feature_type}_start>'
                token_end = f'<{biotype}_{feature_type}_end>'
            else:
                token_start = f'<{feature_type}_start>'
                token_end = f'<{feature_type}_end>'
            tokens_to_insert.append((chrom, start, token_start))
            tokens_to_insert.append((chrom, end + 1, token_end))

    if tokens_to_insert:
        print(tokens_to_insert)
        insert_tokens_and_reset()

    # save the chromosome data
    # save_chrm_to_file(extractor.sequences[chrom],"./root/data/annotated_fasta_improved/test",chrom)
    # return None
    # return the chromosome data
    return (chrom, extractor.sequences[chrom])

def parallel_process_gtf(fasta_file, gtf_file, output_file, num_processes=3):
    """
    Processes a GTF file in parallel and modifies a fasta file based on the GTF data.

    Args:
    fasta_file (str): Path to the fasta file.
    gtf_file (str): Path to the GTF file.
    output_file (str): Path for the output fasta file.
    num_processes (int, optional): Number of processes to use. Default is 3.

    Returns:
    None
    """
    try:
        multiprocessing.log_to_stderr(logging.INFO)

        # Use a manager for creating shared objects
        with multiprocessing.Manager() as manager:
            extractor = NCBIFastaStringExtractor(fasta_file)
            logging.info(f"Processing {gtf_file} with {num_processes} processes")
            print(f"Processing {gtf_file} with {num_processes} processes")

            chromosome_data = split_gtf_file(gtf_file, output_file)
            logging.info(f"Processing {len(chromosome_data)} chromosomes")
            print(f"Processing {len(chromosome_data)} chromosomes")
            tasks = [(chromosome_data[chrom], extractor) for chrom in chromosome_data]
            logging.info(f"Processing {len(tasks)} tasks")
            print(f"Processing {len(tasks)} tasks")

            # Use contextlib.closing to ensure resources are cleaned up properly
            with closing(multiprocessing.Pool(num_processes)) as pool:
                # results = pool.starmap(process_chromosome_data, tasks)
                results = pool.starmap(process_chromosome_data, tasks)
            logging.info("All tasks completed")

        for chrom, sequence in results:
            extractor.update(chrom, sequence)

    except Exception as e:
        logging.error(f"Error in processing: {e}")

    finally:
        # This block ensures that the following code is executed regardless of an exception
        extractor.save_to_file(output_file)
        logging.info(f"Saved to {output_file}")
