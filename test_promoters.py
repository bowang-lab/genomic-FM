from src.datasets.epd_promoters.load_epd import download_epd, parse_epd, species_to_epd

download_epd(out_dir="./root/data/epd")
eukaryote_promoters = get_eukaryote_promoters()
