import json

# Functions to save sequences as JSONl
def save_as_jsonl(data, filename):
    with open(filename, 'w') as f:
        for seq in data:
            f.write(json.dumps(seq)+"\n")