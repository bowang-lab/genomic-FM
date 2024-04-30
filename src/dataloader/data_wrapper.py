from typing import Any
from ..sequence_extractor import GenomeSequenceExtractor, FastaStringExtractor, RandomSequenceExtractor
from ..datasets.clinvar import load_clinvar
from ..datasets.gene_ko.get_gene_knock_out import (
    create_fitness_scores_dataframe,
    create_variant_sequence_and_reference_sequence_for_gene
)
# Import necessary functions from the modules
from ..datasets.cellpassport.load_cell_passport import (
    download_and_extract_cell_passport_file,
    read_vcf,
    extract_cell_line_annotation_from_vcf_file
)
from ..datasets.ensembl_regulatory.load_regulatory import download_regulatory_gff, ensembl_regulatory_species, get_feature_type
from ..datasets.qtl.qtl_loader import process_eqtl_data, process_sqtl_data
from ..datasets.maves.load_maves import get_all_urn_ids, get_score_set, get_scores, get_alternate_dna_sequence
from ..datasets.gwas.load_gwas_catalogue import download_file, extract_snp_details, get_risk_snps, get_summary_stats_for_snp
from ..datasets.olida.load_olida import get_variant_combinations, load_and_process_negative_pairs
from ..datasets.epd_promoters.load_epd import parse_epd, download_epd, species_to_epd
from ..blast_search import run_blast_query
import random
from ..datasets.ncbi_reference_genome.download_ncbi import create_species_taxid_map
from src.datasets.ncbi_reference_genome.get_accession import search_species, get_chromosome_name

from tqdm import tqdm
import random 
import glob 
import os
from pyliftover import LiftOver
import pandas as pd
from kipoiseq import Interval

SPECIES = ['Arabidopsis thaliana', 'Apis mellifera', 'Caenorhabditis elegans', 'Cyprinus carpio carpio', 'Dicentrarchus labra', 'Drosophila melanogaster', 'Danio rerio', 'Gallus gallus', 'Homo sapiens','Macaca mulatta',
           'Mus musculus','Oncorhynchus mykiss', 'Plasmodium falciparum', 'Rattus norvegicus', 'Saccharomyces cerevisiae', 'Salmo salar', 'Schizosaccharomyces pombe', 'Sus scrofa', 'Scophthalmus maximus', 'Zea mays']
ORGANISM = ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum', 'Adrenal_Gland', 'Artery_Aorta', 'Artery_Coronary', 'Artery_Tibial', 'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra',
                        'Breast_Mammary_Tissue', 'Cells_Cultured_fibroblasts', 'Cells_EBV-transformed_lymphocytes', 'Colon_Sigmoid', 'Colon_Transverse', 'Esophagus_Gastroesophageal_Junction', 'Esophagus_Mucosa', 'Esophagus_Muscularis', 'Heart_Atrial_Appendage', 'Heart_Left_Ventricle', 'Kidney_Cortex', 'Liver', 'Lung', 'Minor_Salivary_Gland', 'Muscle_Skeletal', 'Nerve_Tibial', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Small_Intestine_Terminal_Ileum', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
                        'Uterus', 'Vagina', 'Whole_Blood']
        
class OligogenicDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.variant_combinations = get_variant_combinations()
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20):
        # return (x, y) pairs
        data = []

        for variant_combo in tqdm(self.variant_combinations):
            variant_combo_reference = []
            variant_combo_alternate = []
            if variant_combo['FINALmeta'] >= 1:
                try:
                    for variant in ['Variant_1', 'Variant_2']:
                        # Check if key elements are present in the variant data
                        if all(key in variant_combo[variant] for key in ['Chromosome', 'Genomic_Position_Hg38', 'Ref_Allele', 'Alt_Allele']) and \
                            all(variant_combo[variant][key] != "N.A." and not (isinstance(variant_combo[variant][key], float)) for key in ['Chromosome', 'Genomic_Position_Hg38', 'Ref_Allele', 'Alt_Allele']):
                            record = {
                                    'Chromosome': variant_combo[variant]['Chromosome'],
                                    'Position': int(variant_combo[variant]['Genomic_Position_Hg38']),
                                    'Reference Base': variant_combo[variant]['Ref_Allele'],
                                    'Alternate Base': variant_combo[variant]['Alt_Allele'],
                                    'ID': variant_combo['OLIDA_ID']
                                }
                            reference, alternate = self.genome_extractor.extract_sequence_from_record(record, Seq_length)
                            variant_combo_reference.append(reference)
                            variant_combo_alternate.append(alternate)
                        else:
                            raise ValueError("Missing required variant information")

                    variant_combo_reference = 'N'.join(variant_combo_reference)
                    variant_combo_alternate = 'N'.join(variant_combo_alternate)
                    x, y = [variant_combo_reference, variant_combo_alternate, variant_combo['Disease']], 1
                    data.append([x, y])

                except Exception as e:
                    print(f"Skipping variant combination due to error: {e}")
                    continue  

        negative_examples = load_and_process_negative_pairs(Seq_length=Seq_length)
        data += negative_examples
        random.shuffle(data)
        return data

class PromoterDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.epd_promoter_path = download_epd()
        self.species_to_taxids = create_species_taxid_map()
        self.all_records = all_records

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20):
        combined_tuples = []

        for eukaryote_species, species_id in species_to_epd.items():
            if "non-coding" in eukaryote_species:
                eukaryote_species = eukaryote_species.replace(' (non-coding)','')

            print(f"Processing {eukaryote_species}")
            species_dir = os.path.join('./root/data/epd', species_id)
            file_path = next(iter(glob.glob(f"{species_dir}/*.dat")), None)
            
            tax_id = self.species_to_taxids[eukaryote_species]
            fasta_paths = glob.glob(os.path.join("./root/data", tax_id, "ncbi_dataset/data/GC*/GC*fna"))


            if not file_path or not fasta_paths:
                print(f"Required files missing for {eukaryote_species}. Skipping...")
                continue

            fasta_extractor = FastaStringExtractor(fasta_paths[0])
            eukaryote_promoters = parse_epd(file_path)
            promoter_data = []

            for index, promoter in enumerate(tqdm(eukaryote_promoters, desc="Extracting promoters")):
                if not self.all_records and index >= self.num_records:
                    break

                species, gene_name, sequence = promoter['Species'], promoter['Gene Name'], promoter['Sequence'].upper()

                if sequence: 
                    interval = run_blast_query(sequence, fasta_paths[0])
                    if interval is not None:
                        interval = interval.resize(Seq_length)
                        extended_sequence = fasta_extractor.extract(interval)
                        promoter_data.append((species, gene_name, extended_sequence, interval))

            if promoter_data:
                intervals = [data[-1] for data in promoter_data] 
                random_sequences = RandomSequenceExtractor(fasta_paths[0]).extract_random_sequence(
                    length_range=(Seq_length, Seq_length),
                    num_sequences=len(promoter_data),
                    known_regions=intervals
                )
                combined_tuples.extend([[[data[0], data[1], data[2]], 1] for data in promoter_data])
                combined_tuples.extend([[[data[0], data[1], seq], 0] for data, seq in zip(promoter_data, random_sequences)])
            random.shuffle(combined_tuples)

        return combined_tuples

class EnsemblRegulatoryDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.ensembl_regulatory_path = download_regulatory_gff()
        self.all_records = all_records
        self.species_list = ensembl_regulatory_species

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20):
        feature_types = ["enhancer", "TF_binding_site", "CTCF_binding_site", "open_chromatin_region"]
        combined_data = []

        for species in self.species_list:
            print(f"Processing regulatory regions for {species}")
            tax_id = search_species(species)
            fasta_paths = glob.glob(os.path.join("./root/data", tax_id[0], "ncbi_dataset/data/GC*/GC*fna"))

            if not fasta_paths :
                print(f"Required genomic fasta file is missing for {species}. Skipping...")
                continue

            fasta_extractor = FastaStringExtractor(fasta_paths[0])
            ncbi_ids = [key for key in fasta_extractor.fasta.keys() if key.startswith(('NC', 'NL'))]
            ncbi_chromosome_mapping = {}

            for ncbi_id in ncbi_ids:
                chromosome_name = get_chromosome_name(ncbi_id)
                ncbi_chromosome_mapping[ncbi_id] = chromosome_name
            chromosome_ncbi_mapping = {value: key for key, value in ncbi_chromosome_mapping.items()}

            random_extractor = RandomSequenceExtractor(fasta_paths[0])

            for feature_type in feature_types:
                features = get_feature_type(feature_type, species)
                print(f"Processing {len(features)} features of type {feature_type}:")

                feature_data = []
                for index, row in tqdm(features.iterrows()):
                    if not self.all_records and index >= self.num_records:
                        break

                    if row['seqid'] not in chromosome_ncbi_mapping.keys():
                        continue 
                    
                    chrom = chromosome_ncbi_mapping[row['seqid']]
                    start = int(row['start'])
                    end = int(row['end'])
                    interval = Interval(chrom, start, end).resize(Seq_length)

                    # Extract the sequence
                    sequence = fasta_extractor.extract(interval)
                    feature_data.append((species, feature_type, sequence, interval))

                if feature_data:
                    # Generate random sequences, passing intervals to avoid overlapping with known regions
                    intervals = [data[-1] for data in feature_data]
                    random_sequences = random_extractor.extract_random_sequence(
                        length_range=(Seq_length, Seq_length),
                        num_sequences=len(feature_data),
                        known_regions=intervals
                    )

                    combined_data.extend([[[data[0], data[1], data[2]], 1] for data in feature_data])
                    combined_data.extend([[[data[0], data[1], rand_seq], 0] for data, rand_seq in zip(feature_data, random_sequences)])

        return combined_data
        
class MAVEDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.urn_ids = get_all_urn_ids()
        self.all_records = all_records

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='score'):
        # return (x, y) pairs
        data = []
        limit=len(self.urn_ids)

        for urn_id in tqdm(self.urn_ids[:limit], desc="Processing URN IDs"):
            score_set = get_score_set(urn_id)
            for exp in score_set:
                urn_id = exp.get('urn', None)
                title = exp.get('title', None)
                description = exp.get('description', None)
                sequence_type = exp.get('targetGenes', None)[0]['sequence_type']
                annotation = ': '.join([title, description])
                scores = get_scores(urn_id)

                if isinstance(scores, pd.DataFrame) and sequence_type == "dna":
                    if not scores.empty:
                        for index, row in scores.iterrows():
                            if pd.notna(row['hgvs_nt']) and pd.notna(row[target]):
                                reference = exp['targetGenes'][0]['sequence']
                                alternate = get_alternate_dna_sequence(reference, row['hgvs_nt'])
                                
                                if alternate:
                                    if len(reference) <= Seq_length:
                                        x = [reference, alternate, annotation]
                                        y = row[target]
                                        data.append([x,y])
        return data

class GWASDataWrapper:
    def __init__(self, num_records=2000, all_records=True):
        self.num_records = num_records
        self.gwas_catalogue = download_file(file_path='./root/data/gwas_catalog_v1.0.2-associations_e111_r2024-03-01.tsv', gwas_path='alternative')
        self.trait_mappings = download_file(file_path='./root/data/gwas_catalog_trait-mappings_r2024-03-01.tsv', gwas_path='trait_mappings')
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='P-Value'):
        # return (x, y) pairs
        data = []        
        disease_to_efo = self.gwas_trait_mappings.set_index('Disease trait')['EFO term'].to_dict()
        for trait in tqdm(set(disease_to_efo.values())):
            traits = [key for key, value in disease_to_efo.items() if value == trait]
            risk_snps = get_risk_snps(self.gwas_catalog, trait)                
            for index, row in risk_snps.iterrows():
                rsSNP = row['SNPS']
                snp_details = extract_snp_details(self.gwas_catalog, rsSNP, trait)
                if snp_details:
                    summary_stats = get_summary_stats_for_snp(snp_details, trait)
                    if summary_stats:
                        record = {
                            'Chromosome': snp_details['Chromosome'], 
                            'Position': int(snp_details['Position']),
                            'Reference Base': snp_details['Reference'],  
                            'Alternate Base': [snp_details['Risk Allele'][0]],  # Adjust as needed
                            'ID': rsSNP
                        }
                        reference, alternate = self.genome_extractor.extract_sequence_from_record(record, Seq_length)
                        x = [reference, alternate, trait]
                        y = summary_stats[target]
                        data.append([x,y])   
        return data
        
class ClinVarDataWrapper:
    def __init__(self, num_records=30000, all_records=True):
        self.clinvar_vcf_path = load_clinvar.download_file()
        self.records = load_clinvar.read_vcf(self.clinvar_vcf_path,
                                             num_records=num_records,
                                             all_records=all_records)
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def convert_disease_name(self, disease_name):
        """
        Convert a disease name by removing any trailing numbers and underscores.
        For example, converts 'Megacystis-microcolon-intestinal_hypoperistalsis_syndrome_2' to
        'Megacystis-microcolon-intestinal_hypoperistalsis_syndrome'.
        """
        # Split the name by underscores
        parts = disease_name.split('_')

        # Check if the last part is a number, if so, remove it
        if parts[-1].isdigit():
            parts = parts[:-1]

        # Join the parts back together
        return '_'.join(parts)

    def get_data(self, Seq_length=20, target='CLNSIG'):
        # return (x, y) pairs
        data = []
        for record in tqdm(self.records):
            ref,alt = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
            if ref is None:
                continue
            variant_type = record['CLNVC'] # single nucleotide variant, deletion, insertion, etc.
            x = [ref, alt, variant_type]
            if target == 'CLNSIG':
                y = record['CLNSIG'] # benign, likely benign, likely pathogenic, pathogenic, uncertain significance
                if y[0] == "Uncertain_significance" or y[0] == "Conflicting_classifications_of_pathogenicity" or y[0] == "not_provided":
                    continue
                if y[0] == 'Pathogenic/Likely_pathogenic':
                    y[0] = 'Likely_pathogenic'
                if y[0] == 'Benign/Likely_benign':
                    y[0] = 'Likely_benign'
                if y[0] not in ['Benign', 'Likely_benign', 'Likely_pathogenic', 'Pathogenic']:
                    continue
                data.append([x, y[0]])
            elif target == 'CLNDN':
                if record['CLNDN'][0] is not None:
                    y = record['CLNDN'][0].split('|') # disease name
                    # each mutation can be associated with multiple diseases
                    # but most of those diseases are related, so we just take the first one which is not 'not_provided'
                    for disease in y:
                        if disease != 'not_provided':
                            data.append([x, self.convert_disease_name(disease)])
                            break
        return data

class GeneKoDataWrapper:
    def __init__(self, num_records=100, all_records=True):
        self.num_records = num_records
        self.fitness_scores = create_fitness_scores_dataframe()
        self.genome_extractor = GenomeSequenceExtractor()
        if all_records:
            self.num_records = len(self.fitness_scores)
        print(f"Number of records: {self.num_records} out of {len(self.fitness_scores)}")

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, insert_Ns=False):
        # return (x, y) pairs
        data = []
        for i in tqdm(range(self.num_records)):
            gene = self.fitness_scores.iloc[i]
            record = create_variant_sequence_and_reference_sequence_for_gene(gene, insert_Ns=insert_Ns)
            ref, alt = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
            if ref is None:
                continue
            cell_line, cell_line_score = self.flatten(gene)

            x = [ref, alt, cell_line]
            y = cell_line_score
            data.append([x, y])
        return data

    def flatten(self, gene):
        cell_line = gene.index[1:-3]
        cell_line_score = gene[1:-3]
        return list(cell_line), cell_line_score


class CellPassportDataWrapper:
    def __init__(self, num_records=100, all_records=True):
        self.num_records = num_records
        self.cell_passport_files = download_and_extract_cell_passport_file()
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()
        
    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)
    
    def get_data(self, Seq_length=20):
        data = []
        for cell_line_file in tqdm(self.cell_passport_files):
            records = read_vcf(cell_line_file, num_records=self.num_records, all_records=self.all_records)
            for record in records:
                record_types = ['DRV', 'CPV', 'NPGL']
                for record_type in record_types:
                    if record_type in record:
                        ref, alt = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
                        if ref is None:
                            continue
                        annotation = cell_line_file.split('/')[-1].split('.')[0]
                        x = [ref, alt, annotation]
                        y = record_type
                        data.append([x, y])
                        break
        return data


class eQTLDataWrapper:
    def __init__(self, num_records=1000, all_records=True):
        self.num_records = num_records
        self.genome_extractor = GenomeSequenceExtractor()
        self.all_records = all_records
        
    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)
    
    def get_data(self, Seq_length=20, target='slope'):
        data = []
        for organism in tqdm(ORGANISM):
            records = process_eqtl_data(organism=organism)
            if records is None:
                print(f"No records found for {organism}")
            if self.all_records:
                self.num_records = len(records)
            for i in range(self.num_records):
                row = records.iloc[i]
                record = row['record']
                slop = row['slope']
                p_val = row['pval_nominal']
                reference, alternate = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
                if reference is None:
                    continue
                x = [reference, alternate, organism]
                if target == 'slope':
                    y = slop
                elif target == 'p_val':
                    y = p_val
                data.append([x, y])
        return data

class sQTLDataWrapper:
    def __init__(self, num_records=30, all_records=True):
        self.num_records = num_records
        self.genome_extractor = GenomeSequenceExtractor()
        self.all_records = all_records
        
    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)
    
    def get_data(self, Seq_length=20, target='slope'):
        data = []
        for organism in tqdm(ORGANISM):
            records = process_sqtl_data(organism=organism)
            if self.all_records:
                self.num_records = len(records)
            for i in range(self.num_records):
                row = records.iloc[i]
                record = row['record']
                splice_position = row['phenotype_id'].split(':')
                splice_position_distance_change = int(splice_position[2]) - int(splice_position[1])
                slop = row['slope']
                p_val = row['pval_nominal']
                reference, alternate = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
                x = [reference, alternate, organism]
                if target == 'slope':
                    y = slop
                elif target == 'p_val':
                    y = p_val
                elif target == 'splice_change':
                    y = splice_position_distance_change
                data.append([x, y])
        return data
