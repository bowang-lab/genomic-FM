import os
import subprocess
import requests
import gzip

def download_file(file_path='./root/data/dida.zip',
                  gz_path='10749489'):
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"{file_path} not found. Starting download...")

        # URL for the file
        url = f'https://zenodo.org/api/records/{gz_path}/files-archive'

        try:
            # Download the file
            response = requests.get(url)
            compressed_file_path = os.path.join(os.path.dirname(file_path), gz_path)
            with open(compressed_file_path, 'wb') as f:
                f.write(response.content)

            # Unzip the file
            with gzip.open(compressed_file_path, 'rb') as f_in:
                with open(file_path, 'wb') as f_out:
                    f_out.write(f_in.read())


            print(f"File downloaded and unzipped successfully: {file_path}")

        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    else:
        print(f"File already exists: {file_path}")

    return file_path
