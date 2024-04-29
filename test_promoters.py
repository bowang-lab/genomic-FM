from src.datasets.epd_promoters.load_epd import download_epd, get_eukaryote_promoters
from src.utils import save_as_jsonl, read_jsonl
from tqdm import tqdm

download_epd(out_dir="./root/data/epd")
eukaryote_promoters = get_eukaryote_promoters()
save_as_jsonl(eukaryote_promoters,'./root/data/eukaryote_promoters.jsonl')

eukaryote_promoters = read_jsonl('./root/data/eukaryote_promoters.jsonl')
for promoter in tqdm(eukaryote_promoters):
    species, gene, sequence, target = promoter[0], promoter[1], promoter[2], promoter[3]
    x = (species, gene, sequence)
    y = target 