import json

# Functions to save sequences as JSONl
def save_as_jsonl(data, filename):
    with open(filename, 'w') as f:
        for seq in data:
            f.write(json.dumps(seq)+"\n")
            
def read_jsonl(filename):
    """
    Read a JSON Lines file and return a list of dictionaries.
    
    Args:
    filename (str): The path to the JSONL file to read.
    
    Returns:
    List[dict]: A list of dictionaries, each representing one line in the JSONL file.
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                # Parse the JSON data and append to the list
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

