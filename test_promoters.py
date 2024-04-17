from src.datasets.epd.get_epd import download_epd, parse_epd, species_to_epd
import os
import glob

for eukaryote_species, species_id in species_to_epd.items():
    download_epd(species_id, out_dir='./root/data/epd')
    species_dir = os.path.join('./root/data/epd',species_id)
    file_path = glob.glob(f"{species_dir}/*.dat")[0]
    eukaryote_promoters = parse_epd(file_path)
    for promoter in eukaryote_promoters:
        print(promoter)
