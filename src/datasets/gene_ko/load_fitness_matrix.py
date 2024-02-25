import requests
import zipfile
import os
import pandas as pd


def read_tsv_file(filename="root/data/Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_scaled_bayesian_factors_20240111.tsv"):
    """
    Read a TSV file into a Pandas DataFrame, using the second row as the header and
    skipping the first five rows for the rest of the file.

    Parameters:
    filename (str): The path to the TSV file.

    Returns:
    DataFrame: A Pandas DataFrame containing the data from the TSV file, with modifications.
    """
    try:
        # Read the second row for the header
        header = pd.read_csv(filename, sep='\t', skiprows=1, nrows=1).columns

        # Read the rest of the file, skipping the first five rows, and use the extracted header
        df = pd.read_csv(filename, sep='\t', skiprows=5, header=None, names=header)

        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def download_fitness_scores(fitness_url="https://cog.sanger.ac.uk/cmp/download/Project_Score2_fitness_scores_Sanger_v2_Broad_21Q2_20240111.zip", subdirectory="root/data"):
    try:
        # Create the subdirectory if it doesn't exist
        os.makedirs(subdirectory, exist_ok=True)

        # Define the path to the file
        file_path = os.path.join(subdirectory, 'fitness_scores.zip')

        print(f"File path: {file_path}")

        # Check if the file already exists
        if os.path.exists(file_path):
            print("File already exists.")
            return file_path  # Returning the path of the existing file

        # Send a HTTP request to the URL of the file, stream = True prevents the file from
        # being downloaded immediately
        r = requests.get(fitness_url, stream=True)

        # Check if the request is successful
        if r.status_code == 200:
            # Download the file
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)

            # Unzip the file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                print(f"Extracting the file to {subdirectory}...")
                zip_ref.extractall(subdirectory)

            print("File downloaded and unzipped successfully.")
            return os.path.join(subdirectory,
                           'Project_Score_fitness_scores_Sanger_v2_Broad_21Q2_20240111',
                           'Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_scaled_bayesian_factors_20240111.tsv')
        else:
            print("Failed to download the file.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
