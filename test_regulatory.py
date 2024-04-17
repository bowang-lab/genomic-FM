from src.datasets.ensembl_regulatory.get_regulatory import download_regulatory_gff, get_feature_type
from src.sequence_extractor import FastaStringExtractor
from kipoiseq import Interval

# Example usage for Homo sapiens
download_regulatory_gff('Homo sapiens')

# Example usage
enhancers = get_feature_type("enhancer","Homo sapiens")
print(enhancers.head())


SEQUENCE_LENGTH = 1000
genome_extractor = FastaStringExtractor('./root/data/hg38.fa')

for index, row in enhancers.iterrows():
    chrom = "chr" + row['seqid']
    start = row['start']
    end = row['end']

    interval = Interval(chrom, start, end)

    # Use the extractor to get the sequence
    sequence = genome_extractor.extract(interval)

    print(f"{row['ID']}: {sequence}\n")

