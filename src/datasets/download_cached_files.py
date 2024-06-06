import requests
import os

def download_zenodo_files(record_id, save_dir='./root/data'):
    """
    Download all files from a Zenodo record into a specified directory.

    :param record_id: The Zenodo record ID to download files from.
    :param save_dir: The directory where files will be saved. Defaults to './root/data'.
    """
    # Construct the Zenodo API endpoint for the record
    url = f'https://zenodo.org/api/records/{record_id}'

    try:
        # Make a request to get the record data
        response = requests.get(url)
        response.raise_for_status()  # This will raise an error if the request failed

        # Parse the JSON response
        data = response.json()

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Iterate through all files in the record
        for file in data['files']:
            file_url = file['links']['self']
            file_name = file['key']
            file_path = os.path.join(save_dir, file_name)

            # Download each file
            print(f'Downloading {file_name}...')
            file_response = requests.get(file_url)
            file_response.raise_for_status()  # This will raise an error if the request failed

            # Save the file to disk
            with open(file_path, 'wb') as f:
                f.write(file_response.content)

        print('All files downloaded successfully.')

    except requests.RequestException as e:
        print(f"An error occurred: {e}")
