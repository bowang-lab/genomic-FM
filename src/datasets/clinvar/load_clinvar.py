import os
import subprocess
import requests
import gzip
import vcf

def download_file(vcf_file_path='./root/data/clinvar_20240416.vcf',
                  vcf_gz_path='clinvar_20240416.vcf.gz'):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(vcf_file_path), exist_ok=True)

    # Check if the VCF file exists
    if not os.path.exists(vcf_file_path):
        print(f"{vcf_file_path} not found. Starting download...")

        # URL for the VCF file
        url = f'https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/{vcf_gz_path}'

        try:
            # Download the file
            response = requests.get(url)
            compressed_file_path = os.path.join(os.path.dirname(vcf_file_path), vcf_gz_path)
            with open(compressed_file_path, 'wb') as f:
                f.write(response.content)

            # Unzip the file
            with gzip.open(compressed_file_path, 'rb') as f_in:
                with open(vcf_file_path, 'wb') as f_out:
                    f_out.write(f_in.read())

            print(f"File downloaded and unzipped successfully: {vcf_file_path}")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    else:
        print(f"File already exists: {vcf_file_path}")

    return vcf_file_path


def get_info_field(record, field):
    """Helper function to get a field from the INFO dictionary."""
    return record.INFO[field] if field in record.INFO else 'NA'


def read_vcf(vcf_file_path, num_records=5, all_records=False):
    """Read a VCF file and return a list of records as dictionaries.

    Args:
        vcf_file_path (str): Path to the VCF file.
        num_records (int): Number of records to read.

    Returns:
        list: A list of dictionaries, each representing a VCF record.

    Example usage:
        records = read_vcf('path/to/your/vcf_file.vcf')
    """
    vcf_reader = vcf.Reader(open(vcf_file_path, 'r'))

    records = []
    count = 0

    for record in vcf_reader:
        record_data = {
            "Chromosome": record.CHROM,
            "Position": record.POS,
            "ID": record.ID,
            "Reference Base": record.REF,
            "Alternate Base": record.ALT
        }

        info_fields = ['AF_ESP', 'AF_EXAC', 'AF_TGP', 'ALLELEID', 'CLNDN', 'CLNDNINCL', 'CLNDISDB', 'CLNDISDBINCL',
                       'CLNHGVS', 'CLNREVSTAT', 'CLNSIG', 'CLNSIGCONF', 'CLNSIGINCL', 'CLNVC', 'CLNVCSO', 'CLNVI',
                       'DBVARID', 'GENEINFO', 'MC', 'ONCDN', 'ONCDNINCL', 'ONCDISDB', 'ONCDISDBINCL', 'ONC',
                       'ONCINCL', 'ONCREVSTAT', 'ONCCONF', 'ORIGIN', 'RS', 'SCIDN', 'SCIDNINCL', 'SCIDISDB',
                       'SCIDISDBINCL', 'SCIREVSTAT', 'SCI', 'SCIINCL']

        for field in info_fields:
            record_data[field] = get_info_field(record, field)

        records.append(record_data)

        count += 1
        if count >= num_records and not all_records:
            break

    return records
