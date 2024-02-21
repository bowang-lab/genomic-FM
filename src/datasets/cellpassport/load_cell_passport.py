import os
import requests
import gzip
import zipfile
import vcf


def download_and_extract_cell_passport_file(url="https://cog.sanger.ac.uk/cmp/download/mutations_wgs_vcf_20221123.zip",
                                            folder_path='./root/data/',
                                            zip_file_name='mutations_wgs_vcf_20221123.zip'):
    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Path for the downloaded zip file
    zip_file_path = os.path.join(folder_path, zip_file_name)

    # Check if the ZIP file already exists
    if not os.path.exists(zip_file_path):
        # Download the ZIP file
        print(f"Downloading {zip_file_name}...")
        response = requests.get(url)
        with open(zip_file_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"{zip_file_name} already exists.")

    vcf_files = []  # Initialize a list to store the paths of the VCF files

    # Extract files from the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            file_path = os.path.join(folder_path, file)
            if file.endswith('.vcf') or file.endswith('.vcf.gz'):
                # Check if the VCF file or its gzipped version already exists
                if not os.path.exists(file_path):
                    zip_ref.extract(file, folder_path)
                    print(f"Extracted: {file}")

                    # If it's a gzipped VCF, unzip it
                    if file.endswith('.vcf.gz'):
                        with gzip.open(file_path, 'rb') as f_in:
                            unzipped_file_path = file_path[:-3]
                            with open(unzipped_file_path, 'wb') as f_out:
                                f_out.write(f_in.read())
                            print(f"Unzipped: {file}")
                            vcf_files.append(unzipped_file_path)  # Add the unzipped VCF file to the list

                            # Optional: Delete the .gz file
                            # os.remove(file_path)
                    else:
                        vcf_files.append(file_path)  # Add the VCF file to the list
                else:
                    print(f"{file} already exists.")
                    # If it's a gzipped VCF check if the unzipped version exists
                    if file.endswith('.vcf.gz'):
                        unzipped_file_path = file_path[:-3]
                        if not os.path.exists(unzipped_file_path):
                            with gzip.open(file_path, 'rb') as f_in:
                                with open(unzipped_file_path, 'wb') as f_out:
                                    f_out.write(f_in.read())
                            print(f"Unzipped: {file}")
                        else:
                            print(f"{unzipped_file_path} already exists.")
                    # set the vcf file path to the unzipped file path
                    vcf_files.append(unzipped_file_path)  # Add the unzipped VCF file to the list

    # Optional: Delete the downloaded ZIP file
    # os.remove(zip_file_path)

    if vcf_files:
        print("VCF files processed successfully.")
        return vcf_files
    else:
        print("No VCF files found.")
        return []



def get_info_field(record, field):
    """Helper function to get a field from the INFO dictionary."""
    return record.INFO[field] if field in record.INFO else 'NA'


def read_vcf(vcf_file_path, num_records=5, all_records=False, selected_info_fields=None):
    """Read a VCF file and return a list of records as dictionaries, including FORMAT and sample-specific data.

    Args:
        vcf_file_path (str): Path to the VCF file.
        num_records (int): Number of records to read. Ignored if all_records is True.
        all_records (bool): Whether to read all records in the file.
        selected_info_fields (list of str, optional): Specific INFO fields to extract. Extracts all fields if None.

    Returns:
        list: A list of dictionaries, each representing a VCF record.

    Raises:
        FileNotFoundError: If the VCF file does not exist.
        ValueError: If the VCF file is improperly formatted.

    Example usage:
        records = read_vcf('path/to/your/vcf_file.vcf')
    """
    try:
        with open(vcf_file_path, 'r') as vcf_file:
            vcf_reader = vcf.Reader(vcf_file)

            records = []
            count = 0

            for record in vcf_reader:
                record_data = {
                    "Chromosome": record.CHROM,
                    "Position": record.POS,
                    "ID": record.ID,
                    "Reference Base": record.REF,
                    "Alternate Base": record.ALT,
                    "Format": record.FORMAT.split(':')
                }

                info_fields = selected_info_fields if selected_info_fields is not None else record.INFO.keys()
                for field in info_fields:
                    record_data[field] = get_info_field(record, field)

                # Extracting sample-specific data
                samples_data = {}
                for sample in record.samples:
                    sample_data = {}
                    for format_key, format_value in zip(record.FORMAT.split(':'), sample.data):
                        sample_data[format_key] = format_value
                    samples_data[sample.sample] = sample_data

                record_data["Samples"] = samples_data

                records.append(record_data)

                count += 1
                if count >= num_records and not all_records:
                    break

            return records

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {vcf_file_path}")
    except Exception as e:
        raise ValueError(f"Error reading VCF file: {e}")
