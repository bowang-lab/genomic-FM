from .load_gene_position_gtf import CustomAnchoredGTFDl
from .load_fitness_matrix import read_tsv_file, download_fitness_scores
from src.sequence_extractor import FastaStringExtractor
from kipoiseq import Interval
import pandas as pd
import os

def create_fitness_scores_dataframe(fitness_url="https://cog.sanger.ac.uk/cmp/download/Project_Score2_fitness_scores_Sanger_v2_Broad_21Q2_20240111.zip"):
    # check if the file already exists
    if os.path.exists('root/data/fitness_scores.csv'):
        return pd.read_csv('root/data/fitness_scores.csv')
    # Download the annotation and fitness scores
    print("Loading the gene annotations from the GTF file...")
    dl = CustomAnchoredGTFDl(num_upstream=1, num_downstream=1)
    print("Gene annotations loaded.")
    print("Downloading the fitness scores...")
    tsv_path = download_fitness_scores(fitness_url)
    print("Fitness scores downloaded.")
    print("Produce annotations for the fitness scores...")
    fitness_scores = read_tsv_file(tsv_path)
    print(tsv_path)
    print(fitness_scores.head())
    gene_list = fitness_scores['model_id']
    dl.filter_by_gene(gene_list)
    gene_position = get_chroms_pos(dl)
    map_gene_position_to_dataframe(fitness_scores, gene_position)
    fitness_scores = fitness_scores[fitness_scores['chr'].notna() & fitness_scores['anchor_pos'].notna()]
    # Save the chached dataframe
    fitness_scores.to_csv('root/data/fitness_scores.csv', index=False)
    return fitness_scores


def get_chroms_pos(dl):
    chroms_pos = {}
    for line in dl:
        metadata = line['metadata']
        chrom = metadata['ranges'].chr
        anchor_pos = metadata['anchor_pos']
        gene_length = metadata['End'] - metadata['Start']
        gene_name = metadata['gene_name']
        chroms_pos[gene_name] = (chrom, anchor_pos, gene_length)
    return chroms_pos


def map_gene_position_to_dataframe(df, gene_position):
    df['chr'] = df['model_id'].map(lambda x: gene_position.get(x, (None, None, None))[0])
    df['anchor_pos'] = df['model_id'].map(lambda x: gene_position.get(x, (None, None, None))[1])
    df['gene_length'] = df['model_id'].map(lambda x: gene_position.get(x, (None, None, None))[2])
    df['anchor_pos'] = df['anchor_pos'].astype('Int64')
    df['gene_length'] = df['gene_length'].astype('Int64')


def create_variant_sequence_and_reference_sequence_for_gene(gene_name, insert_Ns=False):
    extractor = FastaStringExtractor('root/data/hg38.fa')
    reference_seq = extractor.extract(Interval(gene_name['chr'],
                                               gene_name['anchor_pos'],
                                               gene_name['anchor_pos'] +
                                               gene_name['gene_length']))
    variant_seq = ['']
    if insert_Ns:
        variant_seq = ['N' * gene_name['gene_length']]
    record = {'Chromosome': gene_name['chr'],
              'Position': gene_name['anchor_pos'],
              'Reference Base': reference_seq,
              'Alternate Base': variant_seq,
              'ID': gene_name['model_id']}
    return record
