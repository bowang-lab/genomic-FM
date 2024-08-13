import pandas as pd
import requests
import shutil
import gzip
import os


def download_and_unzip_gtf(output_file,
                           url="https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.basic.annotation.gtf.gz"):
    """
    Download and unzip a GTF file from a given URL.

    :param url: URL of the gzipped GTF file
    :param output_file: Path to save the unzipped GTF file
    """
    # Download the gzipped file
    gz_file = output_file + '.gz'
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(gz_file, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    # Unzip the file
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove the gzipped file
    os.remove(gz_file)
    print(f"Downloaded and unzipped the file to: {output_file}")

def load_gtf(gtf_file="root/gencode.v46.basic.annotation.gtf"):
    """
    Load GTF file into a DataFrame and filter for protein-coding exons.
    """
    if not os.path.exists(gtf_file):
        print("Downloading GTF file of genome annotation...")
        download_and_unzip_gtf(gtf_file)
    gtf_cols = [
        'seqname', 'source', 'feature', 'start', 'end', 'score',
        'strand', 'frame', 'attribute'
    ]
    print(f"Loading GTF file: {gtf_file}")
    gtf = pd.read_csv(
        gtf_file, sep='\t', comment='#', names=gtf_cols, header=None
    )
    # Filter for exons from protein-coding genes or transcripts
    protein_coding = gtf[
        (gtf['feature'] == 'exon') &
        (gtf['attribute'].str.contains('gene_type "protein_coding"') | gtf['attribute'].str.contains('transcript_type "protein_coding"'))
    ]

    return protein_coding

def is_encoding(gtf_df, chrom, start, end):
    """
    Check if a given (chrom, start, end) record is within a protein-coding region.
    """
    # Filter GTF DataFrame for the specific chromosome
    chrom_df = gtf_df[gtf_df['seqname'] == chrom]
    # Check for overlap with any protein-coding exon
    overlapping_exons = chrom_df[
        (chrom_df['start'] <= end) &
        (chrom_df['end'] >= start)
    ]
    return not overlapping_exons.empty



# Check if a specific region is within a protein-coding region
# GTF_DF = load_gtf()
# chrom = 'chr1'
# start = 65420
# end = 71584
# is_in_coding_region = is_encoding(GTF_DF, chrom, start, end)
# print(is_in_coding_region)
