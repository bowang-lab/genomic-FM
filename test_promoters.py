from src.datasets.epd_promoters.load_epd import download_epd, get_eukaryote_promoters

download_epd(out_dir="./root/data/epd")
eukaryote_promoters = get_eukaryote_promoters()
