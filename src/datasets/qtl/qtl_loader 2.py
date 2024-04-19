import requests
import tarfile
import os
import pandas as pd


def download_file(url: str, download_path: str, data_path: str):
    """Download a file from a given URL."""
    print(f"Downloading {url} to {os.path.join(data_path, download_path)}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(os.path.join(data_path, download_path), 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

def extract_tar(file_path: str, extract_to: str):
    """Extract a tar file."""
    with tarfile.open(file_path, 'r') as file:
        file.extractall(extract_to)

def remove_file(file_path: str):
    """Remove a file."""
    if os.path.exists(file_path):
        os.remove(file_path)

def download(url: str,
             data_path: str = "root/data/"):
    """Download eQTL data from GTEx."""
    # Create the data directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Download eQTL data from GTEx
    download_path = "GTEx_Analysis_v8_eQTL.tar"
    download_file(url, download_path, data_path)
    print(f"Downloaded {url} to {os.path.join(data_path, download_path)}")
    # Extract the files
    extract_tar(os.path.join(data_path, download_path), data_path)

    # Remove the tar file
    remove_file(os.path.join(data_path, download_path))

    # Return the path to the extracted files
    return data_path


def process_eqtl_data(organism: str = "Adipose_Subcutaneous",
                      data_path: str = "root/data/"):
    organism_list = ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum', 'Adrenal_Gland', 'Artery_Aorta', 'Artery_Coronary', 'Artery_Tibial', 'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra',
                        'Breast_Mammary_Tissue', 'Cells_Cultured_fibroblasts', 'Cells_EBV-transformed_lymphocytes', 'Colon_Sigmoid', 'Colon_Transverse', 'Esophagus_Gastroesophageal_Junction', 'Esophagus_Mucosa', 'Esophagus_Muscularis', 'Heart_Atrial_Appendage', 'Heart_Left_Ventricle', 'Kidney_Cortex', 'Liver', 'Lung', 'Minor_Salivary_Gland', 'Muscle_Skeletal', 'Nerve_Tibial', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Small_Intestine_Terminal_Ileum', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
                        'Uterus', 'Vagina', 'Whole_Blood']
    assert organism in organism_list, f"Invalid organism, it should be one of {organism_list}"
    """Process eQTL data files for a specified organism in the given directory."""
    if not os.path.exists(os.path.join(data_path, "GTEx_Analysis_v8_eQTL/")):
        # Download the data if it doesn't exist
        download("https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/single-tissue-cis-qtl/GTEx_Analysis_v8_eQTL.tar",
                 data_path=data_path)

    data_path = os.path.join(data_path, "GTEx_Analysis_v8_eQTL")
    # Define the file pattern based on the organism parameter
    file_pattern = f"{organism}.v8.egenes.txt"

    # Search for files matching the pattern
    file = os.path.join(data_path, file_pattern)

    # Check if file exit
    if not os.path.exists(file):
        # check if file.gz exist
        file = os.path.join(data_path, file_pattern + ".gz")
        if not os.path.exists(file):
            print(f"No data files found for organism: {organism}.")
            return
        else:
            df = pd.read_csv(file, sep='\t', compression='gzip')

    df = pd.read_csv(file, sep='\t')
    # Process the DataFrame as needed
    # Additional processing can be done here
    df['record'] = df.apply(lambda x: {'Chromosome': x['chr'],
                                        'Position': x['variant_pos'],
                                        'Reference Base': x['ref'],
                                        'Alternate Base': [x['alt']],
                                        'ID': x['variant_id']}, axis=1)
    return df

def process_sqtl_data(organism: str = "Adipose_Subcutaneous",
                      data_path: str = "root/data/"):
    """Process sQTL data files for a specified organism in the given directory."""
    if not os.path.exists(os.path.join(data_path, "GTEx_Analysis_v8_sQTL/")):
        # Download the data if it doesn't exist
        download("https://storage.googleapis.com/adult-gtex/bulk-qtl/v8/single-tissue-cis-qtl/GTEx_Analysis_v8_sQTL.tar",
                 data_path=data_path)

    data_path = os.path.join(data_path, "GTEx_Analysis_v8_sQTL")
    # Define the file pattern based on the organism parameter
    file_pattern = f"{organism}.v8.sgenes.txt"

    # Search for files matching the pattern
    file = os.path.join(data_path, file_pattern)

    # Check if file exit
    if not os.path.exists(file):
        # check if file.gz exist
        file = os.path.join(data_path, file_pattern + ".gz")
        if not os.path.exists(file):
            print(f"No data files found for organism: {organism}.")
            return
        else:
            df = pd.read_csv(file, sep='\t', compression='gzip')

    df = pd.read_csv(file, sep='\t')
    # Process the DataFrame as needed
    # Additional processing can be done here
    df['record'] = df.apply(lambda x: {'Chromosome': x['chr'],
                                        'Position': x['variant_pos'],
                                        'Reference Base': x['ref'],
                                        'Alternate Base': [x['alt']],
                                        'ID': x['variant_id']}, axis=1)
    return df
