# Import necessary functions from the modules
from src.datasets.dida.dida import download_file, map_digenic_variants
from src.sequence_extractor import GenomeSequenceExtractor
import pandas as pd

# Load the DIDA data

file_path='./root/data/dida'
download_file(dir_path=file_path,
                  record_id='10749489')

# Load the datasets
digenic_variants = pd.read_csv('./root/data/dida/variants.tsv', sep='\t',index_col=False)
variants_full = pd.read_csv('./root/data/dida/variants_full.tsv', sep='\t')


mapped_variants = map_digenic_variants(digenic_variants, variants_full)
print(mapped_variants)
