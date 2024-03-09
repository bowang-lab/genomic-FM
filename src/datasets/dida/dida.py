import os
import subprocess
import requests
import zipfile

def download_file(dir_path='./root/data/dida',
                  record_id='10749489'):
    
    # Create the directory if it does not exist
    os.makedirs(dir_path, exist_ok=True)
    # Construct URL for accessing record metadata
    metadata_url = f'https://zenodo.org/api/records/{record_id}'

    # Make a request to get the record metadata
    response = requests.get(metadata_url)
    response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code.

    # Parse response JSON
    data = response.json()

    # Iterate over all files in the record
    for file_info in data['files']:
        file_url = file_info['links']['self']
        file_name = file_info['key']
        file_path = os.path.join(dir_path, file_name)

        if os.path.exists(file_path):
            print(f"{file_name} already exists. Skipping download.")
            continue
        
        print(f"Downloading {file_name}...")
        # Stream download to handle large files
        with requests.get(file_url, stream=True) as file_response:
            file_response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)

    print("All files have been downloaded.")

