from src.datasets.ensembl_regulatory.load_regulatory import download_regulatory_gff, get_eukaryote_regulatory
from src.utils import save_as_jsonl, read_jsonl
from tqdm import tqdm

download_regulatory_gff(out_dir='./root/data/regulatory_features')
regulatory_regions = get_eukaryote_regulatory()
save_as_jsonl(regulatory_regions,'./root/data/regulatory_features.jsonl')

regulatory_regions = read_jsonl('./root/data/regulatory_features.jsonl')
for regulatory_region in tqdm(regulatory_regions):
    species, feature, sequence, target = regulatory_region[0], regulatory_region[1], regulatory_region[2], regulatory_region[3]
    x = (species, feature, sequence)
    y = target



