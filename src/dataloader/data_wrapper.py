from typing import Any
from ..sequence_extractor import GenomeSequenceExtractor
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
from ..datasets.ensembl_regulatory.get_regulatory import download_regulatory_gff, get_feature_type
from ..datasets.qtl.qtl_loader import process_eqtl_data, process_sqtl_data
from ..datasets.maves.get_maves import get_all_urn_ids, get_score_set, get_scores, get_alternate_sequence
from ..datasets.gwas.gwas_catalogue import download_file, get_trait_mappings, get_unique_risk_snps, extract_snp_details, get_risk_snps
from ..datasets.dida.load_dida import download_file, map_digenic_variants, get_digenic_variants

from pyliftover import LiftOver
import pandas as pd
from kipoiseq import Interval
from tqdm import tqdm


ORGANISM = ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum', 'Adrenal_Gland', 'Artery_Aorta', 'Artery_Coronary', 'Artery_Tibial', 'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra',
                        'Breast_Mammary_Tissue', 'Cells_Cultured_fibroblasts', 'Cells_EBV-transformed_lymphocytes', 'Colon_Sigmoid', 'Colon_Transverse', 'Esophagus_Gastroesophageal_Junction', 'Esophagus_Mucosa', 'Esophagus_Muscularis', 'Heart_Atrial_Appendage', 'Heart_Left_Ventricle', 'Kidney_Cortex', 'Liver', 'Lung', 'Minor_Salivary_Gland', 'Muscle_Skeletal', 'Nerve_Tibial', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Small_Intestine_Terminal_Ileum', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
                        'Uterus', 'Vagina', 'Whole_Blood']
        
class DigenicDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.cell_passport_files = download_and_extract_cell_passport_file()
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='CLNSIG'):
        # return (x, y) pairs
        data = []
        
class EnsemblRegulatoryDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.cell_passport_files = download_and_extract_cell_passport_file()
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='CLNSIG'):
        # return (x, y) pairs
        data = []
        
class MAVEDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.cell_passport_files = download_and_extract_cell_passport_file()
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='CLNSIG'):
        # return (x, y) pairs
        data = []
        
class GWASDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.gwas_catalogue = download_file(file_path='./root/data/gwas_catalog_v1.0.2-associations_e111_r2024-03-01.tsv', gwas_path='alternative')
        self.trait_mappings = download_file(file_path='./root/data/gwas_catalog_trait-mappings_r2024-03-01.tsv', gwas_path='trait_mappings')
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='Value'):
        # return (x, y) pairs
        data = []        

        trait = "height"
        trait_mappings = get_trait_mappings(gwas_catalog, gwas_trait_mappings, trait)
        print(trait_mappings[:10])

        unique_risk_snps = get_unique_risk_snps(gwas_catalog)

        # Display the number of unique SNPs and the first few SNPs as a sample
        print(f"Total unique risk SNPs found: {len(unique_risk_snps)}")

        # Get rsSNPs associated with a trait
        risk_snps = get_risk_snps(gwas_catalog, trait)
        print(risk_snps)

        for index, row in tqdm(risk_snps.iterrows()):
            rsSNP = row['SNPS']

            print(rsSNP)

            # Get information about a rssnp 
            snp_details = extract_snp_details(gwas_catalog, rsSNP, trait)
            print(snp_details)

            record = {
                'Chromosome': snp_details['Chromosome'], 
                'Position': int(snp_details['Position']),
                'Reference Base': snp_details['Reference'],  
                'Alternate Base': [snp_details['Risk Allele'][0]],  # Adjust as needed
                'ID': rsSNP
            }

            reference, alternate = genome_extractor.extract_sequence_from_record(record, SEQUENCE_LENGTH)



class PromoterDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
        self.num_records = num_records
        self.cell_passport_files = download_and_extract_cell_passport_file()
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='CLNSIG'):
        # return (x, y) pairs
        data = []    
        
class ClinVarDataWrapper:
    def __init__(self, num_records=2000, all_records=False):
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
            cell_line, cell_line_score = self.flatten(gene)

            x = [ref, alt, cell_line]
            y = cell_line_score
            data.append([x, y])
            # for i, cell in enumerate(cell_line):
            #     x = [ref, alt, cell]
            #     y = cell_line_score[i]
            #     data.append([x, y])
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