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
from ..datasets.qtl import process_eqtl_data, process_sqtl_data


ORGANISM = ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum', 'Adrenal_Gland', 'Artery_Aorta', 'Artery_Coronary', 'Artery_Tibial', 'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1', 'Brain_Substant',
                        'Breast_Mammary_Tissue', 'Cells_Cultured_fibroblasts', 'Cells_EBV-transformed_lymphocytes', 'Colon_Sigmoid', 'Colon_Transverse', 'Esophagus_Gastroesophageal_Junction', 'Esophagus_Mucosa', 'Esophagus_Muscularis', 'Heart_Atrial_Appendage', 'Heart_Left_Ventricle', 'Kidney_Cortex', 'Liver', 'Lung', 'Minor_Salivary_Gland', 'Muscle_Skeletal', 'Nerve_Tibial', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Small_Intestine_Terminal_Ileum', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
                        'Uterus', 'Vagina', 'Whole_Blood']
class ClinVarDataWrapper:
    def __init__(self, num_records=100):
        self.num_records = num_records
        self.clinvar_vcf_path = load_clinvar.download_file()
        self.records = load_clinvar.read_vcf(self.clinvar_vcf_path, num_records=self.num_records)
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='CLNSIG'):
        # return (x, y) pairs
        data = []
        for record in self.records:
            ref,alt = self.extract_sequence_from_record(record, sequence_length=Seq_length)
            variant_type = record['CLNVC'] # single nucleotide variant, deletion, insertion, etc.
            x = (ref, alt, variant_type)
            if target == 'CLNSIG':
                y = record['CLNSIG'] # benign, likely benign, likely pathogenic, pathogenic, uncertain significance
            elif target == 'CLNDN':
                y = record['CLNDN']  # disease name
            data.append((x, y))
        return data

    def metadata(self):
        # get the statistics of the dataset by variant outcome and disease
        variant_outcome = {}
        disease = {}
        for record in self.records:
            variant_outcome[record['CLNSIG']] = variant_outcome.get(record['CLNSIG'], 0) + 1
            disease[record['CLNDN']] = disease.get(record['CLNDN'], 0) + 1
        return variant_outcome, disease


class GeneKoDataWrapper:
    def __init__(self, num_records=100):
        self.num_records = num_records
        self.fitness_scores = create_fitness_scores_dataframe()
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, insert_Ns=False):
        # return (x, y) pairs
        data = []
        for i in range(self.num_records):
            gene = self.fitness_scores.iloc[i]
            record = create_variant_sequence_and_reference_sequence_for_gene(gene, insert_Ns=insert_Ns)
            ref, alt = self.extract_sequence_from_record(record, sequence_length=Seq_length)
            cell_line, cell_line_score = self.flatten(gene)
            for i, cell in enumerate(cell_line):
                x = (ref, alt, cell)
                y = cell_line_score[i]
                data.append((x, y))
            data.append((x, y))
        return data

    def flatten(self, gene):
        cell_line = gene.index[1:-3]
        cell_line_score = gene[1:-3]
        return cell_line, cell_line_score


class CellPassportDataWrapper:
    def __init__(self, num_records=100):
        self.num_records = num_records
        self.cell_passport_files = download_and_extract_cell_passport_file()
        self.records = read_vcf(self.cell_passport_files[1], num_records=self.num_records)
        self.genome_extractor = GenomeSequenceExtractor()
    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)
    def get_data(self, Seq_length=20):
        #TODO need to define the target
        throw NotImplementedError
        data = []
        for record in self.records:
            ref,alt = self.extract_sequence_from_record(record, sequence_length=Seq_length)
            x = (ref, alt)
        return data


class eQTLDataWrapper:
    def __init__(self, num_records=100):
        self.num_records = num_records
        self.genome_extractor = GenomeSequenceExtractor()
    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)
    def get_data(self, Seq_length=20, target='slope'):
        data = []
        for organism in ORGANISM:
            records = process_eqtl_data(organism=organism)
            row = records.iloc[0]
            record = row['record']
            slop = row['slope']
            p_val = row['pval_nominal']
            reference, alternate = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
            x = (reference, alternate, organism)
            if target == 'slope':
                y = slop
            elif target == 'p_val':
                y = p_val
            data.append((x, y))
        return data

class sQTLDataWrapper:
    def __init__(self, num_records=100):
        self.num_records = num_records
        self.genome_extractor = GenomeSequenceExtractor()
    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)
    def get_data(self, Seq_length=20, target='slope'):
        data = []
        for organism in ORGANISM:
            records = process_sqtl_data(organism=organism)
            row = records.iloc[0]
            record = row['record']
            splice_position = row['phenotype_id'].split(':')
            splice_position_distance_change = splice_position[2] - splice_position[1]
            slop = row['slope']
            p_val = row['pval_nominal']
            reference, alternate = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
            x = (reference, alternate, organism)
            if target == 'slope':
                y = slop
            elif target == 'p_val':
                y = p_val
            elif target == 'splice_change':
                y = splice_position_distance_change
            data.append((x, y))
        return data
