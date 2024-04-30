from src.datasets.epd_promoters.load_epd import download_epd, get_eukaryote_promoters
from src.utils import save_as_jsonl, read_jsonl

# Example usage
download_epd(out_dir="./root/data/epd")
eukaryote_promoters = get_eukaryote_promoters()
save_as_jsonl(eukaryote_promoters,'./root/data/eukaryote_promoters.jsonl')