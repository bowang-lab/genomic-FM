from src.datasets.maves.load_maves import get_maves
from src.utils import save_as_jsonl, read_jsonl

# Example usage
maves = get_maves()
save_as_jsonl(maves,'./root/data/maves.jsonl')
