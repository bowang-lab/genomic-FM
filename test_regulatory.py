from src.datasets.ensembl_regulatory.load_regulatory import download_regulatory_gff, get_eukaryote_regulatory
from src.utils import save_as_jsonl

download_regulatory_gff(out_dir='./root/data/regulatory_features')
regulatory_regions = get_eukaryote_regulatory()
save_as_jsonl(regulatory_regions,'./root/data/regulatory_regions.jsonl')
