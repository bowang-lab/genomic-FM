from src.datasets.ensembl_regulatory.load_regulatory import download_regulatory_gff, get_regulatory_regions

download_regulatory_gff(out_dir='./root/data/regulatory_features')
regulatory_regions = get_regulatory_regions()


