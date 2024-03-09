# Import necessary functions from the modules
from src.datasets.dida.dida import download_file
from src.sequence_extractor import GenomeSequenceExtractor
import pandas as pd

# Load the GWAS catalog data

file_path='./root/data/dida'
download_file(dir_path=file_path,
                  record_id='10749489')

# Load the datasets
variant_combinations = pd.read_csv('./root/data/dida/variantcombinations.tsv', sep='\t')
variants_full = pd.read_csv('./root/data/dida/variants_full.tsv', sep='\t')


