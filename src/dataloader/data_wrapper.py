from typing import Any, List, Tuple, Union
from pathlib import Path
import glob
import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

from ..utils import save_as_jsonl, read_jsonl
from ..sequence_extractor import GenomeSequenceExtractor, FastaStringExtractor, RandomSequenceExtractor
from ..datasets.clinvar import load_clinvar
from ..datasets.gene_ko.get_gene_knock_out import (
    create_fitness_scores_dataframe,
    create_variant_sequence_and_reference_sequence_for_gene
)
from ..datasets.cellpassport.load_cell_passport import (
    download_and_extract_cell_passport_file,
    read_vcf,
    extract_cell_line_annotation_from_vcf_file
)
from ..datasets.qtl.qtl_loader import process_eqtl_data, process_sqtl_data
from ..datasets.maves.load_maves import get_maves
from ..datasets.gwas.load_gwas_catalogue import download_file, extract_snp_details, get_risk_snps, get_summary_stats_for_snp
from ..datasets.olida.load_olida import get_variant_combinations, load_and_process_negative_pairs
from ..datasets.verified_GV.load_real_clinvar import load_real_clinvar
from ..datasets.maves.mave_utils import MAVE_METHODS, expand_method_filters
import pandas as pd

SPECIES = ['Arabidopsis thaliana', 'Apis mellifera', 'Caenorhabditis elegans', 'Cyprinus carpio carpio', 'Dicentrarchus labra', 'Drosophila melanogaster', 'Danio rerio', 'Gallus gallus', 'Homo sapiens','Macaca mulatta',
           'Mus musculus','Oncorhynchus mykiss', 'Plasmodium falciparum', 'Rattus norvegicus', 'Saccharomyces cerevisiae', 'Salmo salar', 'Schizosaccharomyces pombe', 'Sus scrofa', 'Scophthalmus maximus', 'Zea mays']
ORGANISM = ['Whole_Blood']
CELL_LINE = 0
# ORGANISM = ['Adipose_Subcutaneous', 'Adipose_Visceral_Omentum', 'Adrenal_Gland', 'Artery_Aorta', 'Artery_Coronary', 'Artery_Tibial', 'Brain_Amygdala', 'Brain_Anterior_cingulate_cortex_BA24', 'Brain_Caudate_basal_ganglia', 'Brain_Cerebellar_Hemisphere', 'Brain_Cerebellum', 'Brain_Cortex', 'Brain_Frontal_Cortex_BA9', 'Brain_Hippocampus', 'Brain_Hypothalamus', 'Brain_Nucleus_accumbens_basal_ganglia', 'Brain_Putamen_basal_ganglia', 'Brain_Spinal_cord_cervical_c-1', 'Brain_Substantia_nigra',
                        # 'Breast_Mammary_Tissue', 'Cells_Cultured_fibroblasts', 'Cells_EBV-transformed_lymphocytes', 'Colon_Sigmoid', 'Colon_Transverse', 'Esophagus_Gastroesophageal_Junction', 'Esophagus_Mucosa', 'Esophagus_Muscularis', 'Heart_Atrial_Appendage', 'Heart_Left_Ventricle', 'Kidney_Cortex', 'Liver', 'Lung', 'Minor_Salivary_Gland', 'Muscle_Skeletal', 'Nerve_Tibial', 'Ovary', 'Pancreas', 'Pituitary', 'Prostate', 'Skin_Not_Sun_Exposed_Suprapubic', 'Skin_Sun_Exposed_Lower_leg', 'Small_Intestine_Terminal_Ileum', 'Spleen', 'Stomach', 'Testis', 'Thyroid',
                        # 'Uterus', 'Vagina', 'Whole_Blood']
# Global variable for disease subset filtering (set dynamically via set_disease_subset_from_file)
DISEASE_SUBSET = None

def set_disease_subset_from_file(file_path):
    """Load disease subset from a text file (one disease per line) and set DISEASE_SUBSET global variable."""
    global DISEASE_SUBSET
    try:
        with open(file_path, 'r') as f:
            DISEASE_SUBSET = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(DISEASE_SUBSET)} diseases from {file_path}")
        return DISEASE_SUBSET
    except Exception as e:
        print(f"Warning: Could not load disease subset file {file_path}: {e}")
        DISEASE_SUBSET = None
        return None

DISEASE_SUBSET_heart = ['Familial_thoracic_aortic_aneurysm_and_aortic_dissection', 'Isolated_thoracic_aortic_aneurysm', 'Left_ventricular_noncompaction', 'Primary_dilated_cardiomyopathy', 'Left_ventricular_noncompaction_cardiomyopathy', 'Familial_restrictive_cardiomyopathy', 'Cardiomyopathy', 'PRDM16-related_congenital_heart_disease', 'Long_QT_syndrome', 'Atrial_fibrillation', 'Atrial_standstill', 'Cardiac_arrhythmia', 'Coronary_artery_disorder', 'Dilated_Cardiomyopathy', 'Hypertension', 'Cardiovascular_phenotype', 'Hypertrophic_cardiomyopathy', 'Primary_familial_hypertrophic_cardiomyopathy', 'Idiopathic_cardiomyopathy', 'Rheumatic_heart_disease', 'Congenital_aneurysm_of_ascending_aorta', 'Aortic_aneurysm', 'Myocardial_infarction', 'Atrial_conduction_disease', 'Arrhythmogenic_right_ventricular_dysplasia', 'Dilated_cardiomyopathy_1CC', 'Dilated_cardiomyopathy_1A', 'Primary_familial_dilated_cardiomyopathy', 'Cardioacrofacial_dysplasia', 'sporadic_abdominal_aortic_aneurysm', 'Brugada_syndrome', 'Catecholaminergic_polymorphic_ventricular_tachycardia', 'Familial_atrial_fibrillation', 'Gaucher_disease-ophthalmoplegia-cardiovascular_calcification_syndrome', 'Cardio-facio-cutaneous_syndrome', 'Paroxysmal_familial_ventricular_fibrillation', 'Dilated_cardiomyopathy-hypergonadotropic_hypogonadism_syndrome', 'Heart-hand_syndrome', 'Conduction_system_disorder', 'Right_ventricular_cardiomyopathy', 'Premature_coronary_artery_atherosclerosis', 'Dilated_cardiomyopathy_1D', 'Dilated_cardiomyopathy_1DD', 'TNNT2-Related_Cardiomyopathy', 'Sudden_cardiac_death', 'Dilated_cardiomyopathy_1S', 'Myocarditis', 'TNNT2_-related_cardiomyopathies', 'Hypokinetic_non-dilated_cardiomyopathy', 'Left_ventricular_hypertrophy', 'Periventricular_leukomalacia', 'Atrial_septal_defect', 'Dilated_cardiomyopathy_1V', 'Periventricular_nodular_heterotopia', 'Essential_hypertension', 'Cardioencephalomyopathy', 'Dilated_cardiomyopathy_1AA', 'Intrinsic_cardiomyopathy', 'Syncope', 'Ventricular_arrhythmias_due_to_cardiac_ryanodine_receptor_calcium_release_deficiency_syndrome', 'Conduction_disorder_of_the_heart', 'Cardiac_arrest', 'Arrhythmogenic_right_ventricular_cardiomyopathy', 'Ventricular_fibrillation', 'Pericardial_effusion', 'Early-onset_coronary_artery_disease', 'Pulmonary_hypertension', 'Pulmonary_arterial_hypertension', 'Coronary_artery_disease', 'Congenital_heart_disease', 'Heart_defect_-_tongue_hamartoma_-_polysyndactyly_syndrome', 'Portal_hypertension', 'Neurodevelopmental-craniofacial_syndrome_with_variable_renal_and_cardiac_abnormalities', 'Ascending_aortic_dissection', 'Coronary_heart_disease', 'Early-onset_myopathy_with_fatal_cardiomyopathy', 'Dilated_cardiomyopathy_1G', 'Ventricular_tachycardia', 'Supraventricular_tachycardia', 'Family_history_of_cardiomyopathy', 'Restrictive_cardiomyopathy', 'Interstitial_cardiac_fibrosis', 'Congestive_heart_failure', 'Premature_ventricular_contraction', 'Third_degree_atrioventricular_block', 'CAP-congenital_myopathy_with_arthrogryposis_multiplex_congenita_without_heart_involvement', 'Heart_failure', 'Noncompaction_cardiomyopathy', 'Systolic_heart_failure', 'Aortic_dissection', 'Primary_pulmonary_hypertension', 'Idiopathic_and/or_familial_pulmonary_arterial_hypertension', 'Pulmonary_arterial_hypertension_associated_with_congenital_heart_disease', 'Drug-_or_toxin-induced_pulmonary_arterial_hypertension', 'Pulmonary_arterial_hypertension_associated_with_another_disease', 'Dilated_cardiomyopathy_1I', 'Progressive_familial_heart_block', 'Coarctation_of_aorta', 'Heart_block', 'Short_QT_syndrome', 'Infantile_hypertrophic_cardiomyopathy_due_to_MRPL44_deficiency', 'Abnormal_cardiovascular_system_morphology', 'Congenital_long_QT_syndrome', 'Atrioventricular_septal_defect', 'Congenital_heart_defects', 'Ventricular_septal_defect', 'Dilated_cardiomyopathy_1NN', 'Severe_hypotonia-psychomotor_developmental_delay-strabismus-cardiac_septal_defect_syndrome', 'Chronic_atrial_and_intestinal_dysrhythmia', 'Familial_atrioventricular_septal_defect', 'Brugada_syndrome_(shorter-than-normal_QT_interval)', 'Dilated_cardiomyopathy_1E', 'Familial_isolated_arrhythmogenic_right_ventricular_dysplasia', 'Acquired_long_QT_syndrome', 'Atrioventricular_block', 'Sudden_cardiac_arrest', 'Cardiac_conduction_defect', 'Dilated_cardiomyopathy_1Z', 'Sinoatrial_node_dysfunction_and_deafness', 'Heart', 'Familial_aortic_aneurysms', 'Cardiac_valvular_defect', 'Mitochondrial_DNA_depletion_syndrome_14_(cardioencephalomyopathic_type)', 'Congestive_heart_failure_and_beta-blocker_response', 'Idiopathic_pulmonary_arterial_hypertension', 'Sudden_cardiac_failure', 'Right_aortic_arch', 'Mitochondrial_DNA_depletion_syndrome_12B_(cardiomyopathic_type)', 'Mitochondrial_DNA_depletion_syndrome_12A_(cardiomyopathic_type)', 'Family_history_of_sudden_cardiac_death', 'Familial_cardiomyopathy', 'Dilated_cardiomyopathy_1GG', 'Cardiac_valvular_dysplasia', 'ADAMTS19-associated_congenital_heartdefect', 'Immunodeficiency_93_and_hypertrophic_cardiomyopathy', 'Cognitive_impairment_-_coarse_facies_-_heart_defects_-_obesity_-_pulmonary_involvement_-_short_stature_-_skeletal_dysplasia_syndrome', 'Hypoplastic_left_heart_syndrome', 'Dilated_cardiomyopathy_1L', 'Conotruncal_heart_malformations', 'Malformation_of_the_heart_and_great_vessels', 'Arrhythmogenic_cardiomyopathy_with_wooly_hair_and_keratoderma', 'Arrhythmogenic_cardiomyopathy', 'DSP-related_cardiomyopathy', 'Coronary_artery_atherosclerosis', 'CAP2-associated_dilated_cardiomyopathy', 'Hypoplastic_aortic_arch', 'Histiocytoid_cardiomyopathy', 'Mitochondrial_hypertrophic_cardiomyopathy_with_lactic_acidosis_due_to_MTO1_deficiency', 'Cardiospondylocarpofacial_syndrome', 'Dilated_cardiomyopathy_1JJ', 'Dilated_cardiomyopathy_3B', 'Dilated_cardiomyopathy_1P', 'Atrioventricular_septal_defect_and_common_atrioventricular_junction', 'Coronary_sclerosis', 'Dilated_cardiomyopathy_1J', 'Aortic_valve_disease', 'Supravalvar_aortic_stenosis', 'unspecified_heart_condition', 'Dilated_cardiomyopathy_2B', 'Orthostatic_hypotension', 'Disorder_of_cardiovascular_system', 'FLNC-associated_cardiomyopathy', 'Two-raphe_bicuspid_aortic_valve', 'Cardiofaciocutaneous_syndrome', 'Familial_cardiofaciocutaneous_syndrome', 'Short_QT_syndrome_type', 'Prolonged_QT_interval', 'Long_QT_syndrome_1/2', 'Coronary_artery_spasm', 'Familial_Hypertrophic_Cardiomyopathy_with_Wolff-Parkinson-White_Syndrome', 'Lethal_congenital_glycogen_storage_disease_of_heart', 'Autosomal_dominant_slowed_nerve_conduction_velocity', 'Testicular_anomalies_with_or_without_congenital_heart_disease', 'Autosomal_dominant_intellectual_disability-craniofacial_anomalies-cardiac_defects_syndrome', 'Congenital_heart_disease_(variable)', 'Intellectual_disability-cardiac_anomalies-short_stature-joint_laxity_syndrome', 'Hypotension', 'Three_Vessel_Coronary_Disease', 'Recurrent_metabolic_encephalomyopathic_crises-rhabdomyolysis-cardiac_arrhythmia-intellectual_disability_syndrome', 'Thoracic_aortic_aneurysm', 'Dilated_cardiomyopathy_1X', 'Pulmonary_arterial_hypertension_related_to_hereditary_hemorrhagic_telangiectasia', 'Neonatal_encephalomyopathy-cardiomyopathy-respiratory_distress_syndrome', 'Congenital_heart_defects_and_skeletal_malformations_syndrome', 'ABL1-related_congenital_heart_defects_and_skeletal_malformations_syndrome', 'Aortic_dilatation', 'Congenital_heart_anomalies', 'Aortic_valve_disorder', 'Neurooculocardiogenitourinary_syndrome', 'Fetal_Cardiomyopathy', 'Immunodeficiency_80_with_or_without_congenital_cardiomyopathy', 'Short_QT_Syndrome', 'NEBL-related_Cardiomyopathy', 'Dilated_cardiomyopathy_1KK', 'congenital_heart_defects', 'Dilated_cardiomyopathy_1W', 'Aborted_sudden_cardiac_death', 'Dilated_cardiomyopathy_1C', 'Familial_hypertrophic_cardiomyopathy', 'Isolated_Noncompaction_of_the_Ventricular_Myocardium', 'Familial_isolated_dilated_cardiomyopathy', 'Aneurysm_of_descending_aorta', 'Descending_aortic_dissection', 'ANKRD1-related_dilated_cardiomyopathy', 'Acute_coronary_syndrome', 'Familial_dilated_cardiomyopathy_and_peripheral_neuropathy', 'Dilated_cardiomyopathy_1HH', 'Dilated_cardiomyopathy_1M', 'Paroxysmal_atrial_fibrillation', 'MYBPC3-related_cardiomyopathies', 'Cardiac-urogenital_syndrome', 'Heritable_Thoracic_Aortic_Disease', 'Dilated_cardiomyopathy_1II', 'Fatal_infantile_mitochondrial_cardiomyopathy', 'C1QTNF5-related_condition', 'Bicuspid_aortic_valve', 'Long_qt_syndrome', 'Concentric_hypertrophic_cardiomyopathy', 'Hepatorenocardiac_degenerative_fibrosis', 'Heart_disease', 'Brachydactyly-arterial_hypertension_syndrome', 'Dilated_cardiomyopathy_1O', 'Arrhythmogenic_ventricular_cardiomyopathy', 'Neurodevelopmental_disorder_with_cardiomyopathy', 'Dilated_cardiomyopathy_1T', 'Cardiomyopathy-hypotonia-lactic_acidosis_syndrome', 'Neurocardiofaciodigital_syndrome', 'Cardiac_anomalies_-_developmental_delay_-_facial_dysmorphism_syndrome', 'Impaired_intellectual_development_and_distinctive_facial_features_with_cardiac_defects', 'Neurodevelopmental_disorder_with_brain_anomalies_and_with_or_without_vertebral_or_cardiac_anomalies', 'Lethal_left_ventricular_non-compaction-seizures-hypotonia-cataract-developmental_delay_syndrome', 'Dilated_cardiomyopathy_1FF', 'Intraventricular_hemorrhage', 'Dilated_cardiomyopathy_1EE', 'MYH6-related_cardiac_defects', 'Atrial_flutter', 'Biventricular_noncompaction_cardiomyopathy', 'Congenital_heart_defects_and_ectodermal_dysplasia', 'Structural_heart_defects_and_renal_anomalies_syndrome', 'Dilated_cardiomyopathy_1U', 'Dilated_cardiomyopathy_1R', 'Cardiac_malformation', 'Cardiofacioneurodevelopmental_syndrome', 'Marfan_Syndrome/Loeys-Dietz_Syndrome/Familial_Thoracic_Aortic_Aneurysms_and_Dissections', 'Acute_aortic_dissection', 'Thoracic_aortic_disease', 'Gnb5-related_intellectual_disability-cardiac_arrhythmia_syndrome', 'Dilated_cardiomyopathy_1Y', 'Pulmonary_atresia_with_intact_ventricular_septum', 'Effort-induced_polymorphic_ventricular_tachycardia', 'Sinoatrial_node_disorder', 'Neonatal_cardiomyopathy', 'Acute_myocardial_infarction', 'NR2F2-related_congenital_heart_defects', 'Coronary_artery_disease/myocardial_infarction', 'Cardiac', 'Laterality_defect_and_complex_congenital_heart_disease', 'TRAF7-associated_heart_defect-digital_anomalies-facial_dysmorphism-motor_and_speech_delay_syndrome', 'MYLK3-associated_cardiomyopathy', 'Encephalopathy-hypertrophic_cardiomyopathy-renal_tubular_disease_syndrome', 'Preterm_intraventricular_hemorrhage', 'Abnormal_cardiac_atrium_morphology', 'Atypical_coarctation_of_aorta', 'Neurodevelopmental_disorder_with_relative_macrocephaly_and_with_or_without_cardiac_or_endocrine_anomalies', 'Familial_atrial_myxoma', 'Intellectual_developmental_disorder_with_cardiac_defects_and_dysmorphic_facies', 'Pancreatic_hypoplasia-diabetes-congenital_heart_disease_syndrome', 'ARRHYTHMOGENIC_RIGHT_VENTRICULAR_DYSPLASIA', 'Dilated_cardiomyopathy_1BB', 'SMAD2-related_cardiac_disorders', 'SMAD2-congenital_heart_disease_and_multiple_congenital_anomaly_disorder', 'Periventricular_nodular_heterotopia_with_syndactyly', 'Choanal_atresia-hearing_loss-cardiac_defects-craniofacial_dysmorphism_syndrome', 'Right_atrial_isomerism', 'Heart_and_brain_malformation_syndrome', 'Arrhythmogenic_cardiomyopathy_with_variable_ectodermal_abnormalities', 'PPP1R13L-associated_cardiac_phenotype', 'Cardio-cutaneous_syndrome', 'Glycogen_storage_disease_due_to_muscle_and_heart_glycogen_synthase_deficiency', 'Progressive_familial_heart_block_type_IB', 'Dilated_cardiomyopathy_2A', 'Isolated_Nonsyndromic_Congenital_Heart_Disease', 'Periventricular_heterotopia_with_microcephaly', 'Periventricular_laminar_heterotopia', 'Long_QT_syndrome_2/5', 'Velocardiofacial_syndrome', 'CELSR1-associated_congenital_heartdefects', 'Fatal_Infantile_Cardioencephalomyopathy', 'Oculofaciocardiodental_syndrome', 'Periventricular_nodular_heterotopia_and_epilepsy', 'Uruguay_Faciocardiomusculoskeletal_syndrome', 'FLNA-related_periventricular_nodular_heterotopia', 'X-linked_intellectual_disability-cardiomegaly-congestive_heart_failure_syndrome', 'Encephalocardiomyopathy', 'Abnormal_aortic_valve_physiology']
DISEASE_SUBSET_Original = ['Laterality_defect_', 'Autosomal_dominant_intellectual_disability-craniofacial_anomalies-cardiac_defects_syndrome', 'Infant_onset_multiple_organ_failure', 'ATP2B4-related_condition', 'Hypokinetic_non-dilated_cardiomyopathy', 'Progressive_familial_heart_block_type_IB', 'Dilated_cardiomyopathy_1II', 'Cognitive_impairment_-_coarse_facies_-_heart_defects_-_obesity_-_pulmonary_involvement_-_short_stature_-_skeletal_dysplasia_syndrome', 'Heart-h', 'Vascular_dilatation', 'LDLR-related_condition', 'SHORT_SLEEP', 'HYPERLIPOPROTEINEMIA', 'PIK3CG-related_condition', 'MEF2C_Haploinsufficiency_Syndrome', 'Debrisoquine', 'APOLIPOPROTEIN_A-I_(NORWAY)', 'Coronary_artery_spasm', 'PRKCD-related_condition', 'Combined_low_LDL_and_fibrinogen', 'APOE4_VARIANT', 'Pseudoxanthoma_elasticum', 'TNNT2-related_condition', 'Prolonged_QT_interval', 'ABCA1-related_condition', 'HEMOGLOBIN_G_(TAIWAN-AMI)', 'HCN2_related_developmental_', 'Biventricular_noncompaction_cardiomyopathy', 'MYH2-related_myopathy', 'Cardiac_anomalies_-_developmental_delay_-_facial_dysmorphism_syndrome', 'Glycogen_storage_disease_due_to_muscle_', 'ADCK3-Related_Disorders', 'Vasculitis_due_to_ADA2_deficiency', 'TPM1-related_condition', 'Arterial_tortuosity_syndrome', 'TECRL-related_condition', 'HDLBP-related_condition', 'MRTFB-related_condition', 'Chromosome_1q21.1_deletion_syndrome', 'NDUFA9-related_condition', 'Isolated_Noncompaction_of_the_Ventricular_Myocardium', 'Tuberous_sclerosis_syndrome', 'ITGA8-related_condition', 'Myopathy_caused_by_variation_in_POMT1', 'ANK3-related_condition', 'HOMER2-related_condition', 'Descending_aortic_dissection', 'PRKAA2-related_condition', 'Arterial_dissection', 'Exercise_intolerance_and_complex_III_deficiency', 'ANK3-related_disorder', '_disease_type_2A', 'USP53-related_condition', 'GALNT10-related_condition', 'SCN3A-_Related_Disorder', 'Noonan-like_disorder', 'TNFRSF4-related_condition', 'GJA1-related_condition', 'Mycotic_Aneurysm', 'DLL1-related_condition', 'Mitochondrial_complex_V_(ATP_synthase)_deficiency', 'QRSL1-related_condition', 'NPR2-Related_Disorders', 'Glycogen_storage_disease_due_to_muscle_and_heart_glycogen_synthase_deficiency', 'KCNH7-related_condition', 'Vascular_Malformations_and_Overgrowth', 'methamphetamine_use_disorder', 'KCNH2-related_condition', 'Impaired_intellectual_development_and_distinctive_facial_features_with_cardiac_defects', 'NEUROPEPTIDE_Y_POLYMORPHISM', 'FLNC-Related_Disorders', 'TBX10-related_condition', '_ptosis', 'Arrhythmogenic_cardiomyopathy_with_wooly_hair_', 'GATA5-related_condition', 'Multisystemic_smooth_muscle_dysfunction_syndrome', 'Koolen-de_Vries_syndrome', 'Childhood_myocerebrohepatopathy_spectrum', 'KCNJ5-related_condition', 'Glycogen_storage_disease_IXa2', 'Anemia', 'CAV3-related_condition', 'Capillary_malformation-arteriovenous_malformation', 'RRM2B-related_condition', 'Neurodevelopmental_disorder_with_brain_anomalies_and_with_or_without_vertebral_or_cardiac_anomalies', 'Fibrosis', 'Neurodevelopmental_disorder_with_epilepsy_and_hemochromatosis', 'P2RX1-related_condition', 'Dilated_cardiomyopathy_1S', 'LZTFL1-related_condition', 'Glycogen_storage_disease_type_X', 'Pseudohypoaldosteronism_type_2A', 'OTOF-related_condition', 'PRKCB-related_condition', 'Dilated_cardiomyopathy_1T', 'SIX3-related_condition', 'Capillary_leak_syndrome', 'Periventricular_laminar_heterotopia', 'Polymicrogyria_with_or_without_vascular-type_Ehlers-Danlos_syndrome', 'Congenital_heart_anomalies', 'Pericardial_effusion', 'Glycogen_storage_disease_XV', 'Tetralogy_of_Fallot', 'Sox17-_related_disorders', 'CRYAB-related_condition', 'Conotruncal_heart_malformations', 'PLXNA4-related_condition', 'MYH7-related_skeletal_myopathy', 'Bamforth-Lazarus_syndrome', 'Carney_complex', 'Marfan_Syndrome/Loeys-Dietz_Syndrome/Familial_Thoracic_Aortic_Aneurysms_and_Dissections', 'SEMA3A-related_condition', 'Muscular_dystrophy-dystroglycanopathy_(limb-girdle)', 'Congenital_diarrhea', 'Qualitative_or_quantitative_defects_of_dystrophin', 'Cardiofaciocutaneous_syndrome', 'FBN1-related_condition', 'KCNK3-related_condition', 'PKD1L1-related_condition', 'Abnormality_of_the_eye', 'Bardet-biedl_syndrome_1/2', 'Structural_heart_defects_and_renal_anomalies_syndrome', 'Polycystic_kidney_disease', 'GYS1-related_condition', 'Muscular_channelopathy', 'TRPM4-related_condition', 'XIRP2-related_condition', 'Glycogen_storage_disease_type_III', 'KCNS3-related_condition', 'Wooly_hair-palmoplantar_keratoderma_syndrome', 'Rheumatic_heart_disease', 'HCN1-Related_disorders', 'NDUFAF2-Related_Disorders', 'NT5E-related_condition', '_Dissections', 'Abnormal_mitral_valve_physiology', 'RBM20-related_condition', 'PDE2A-related_condition', 'Fatal_infantile_hypertonic_myofibrillar_myopathy', 'Long_chain_3-hydroxyacyl-CoA_dehydrogenase_deficiency', 'Abnormal_cardiac_atrium_morphology', 'Abnormality_of_connective_tissue', 'McKusick-Kaufman_syndrome', 'TAFAZZIN-related_condition', 'SMAD3-Related_Disorder', 'GATA4-related_condition', 'Venous_malformation', 'Noonan_syndrome_with_multiple_lentigines', 'Familial_hypokalemia-hypomagnesemia', '_autonomic_neuropathy', 'TLR4_POLYMORPHISM', 'NRG1-related_condition', 'AHNAK2-related_condition', 'GDF1-related_condition', 'NDUFA11-related_condition', 'Dermatofibrosarcoma_protuberans', 'Atrioventricular_septal_defect_and_common_atrioventricular_junction', '_distal_skeletal_defects', 'GNB3_POLYMORPHISM', 'AMYLOIDOSIS', 'Ventriculomegaly-cystic_kidney_disease', 'KCNE2-related_condition', 'Susceptibility_to_severe_coronavirus_disease_(COVID-19)_due_to_high_plasma_levels_of_TNF', 'ERLIN1-related_condition', 'CD36-related_condition', 'Arteriovenous_malformation', 'Increased_nuchal_translucency', 'APC-related_condition', 'PKD1-related_condition', 'Aneurysm_of_descending_aorta', 'Chuvash_polycythemia', 'SIRT1-related_condition', 'COX6A2-related_condition', 'Bardet-biedl_syndrome_2/4', 'Renal_hypoplasia', 'Ehlers-Danlos_syndrome_progeroid_type', 'Gastrointestinal_defects_and_immunodeficiency_syndrome', 'Conduction_system_disorder', '_motor_dysfunction', '_lactic_acidosis', 'Systemic_mast_cell_disease', 'Laminopathy', 'SIX2-related_condition', '_ID', 'TLL1-related_condition', 'G6PC3-related_condition', 'Plasma_triglyceride_level_quantitative_trait_locus', 'LIPG-related_condition', 'KCNA5-related_condition', 'Very_long_chain_acyl-CoA_dehydrogenase_deficiency', 'Intellectual_Disability_with_multiple_congenital_anomalies', 'Acquired_long_QT_syndrome', 'Oculofaciocardiodental_syndrome', 'KCND2-related_condition', 'Cardiovascular_phenotype', 'CCL2-related_condition', 'TNNT2-related_disorder', 'Insulin-resistant_diabetes_mellitus_AND_acanthosis_nigricans', 'Partial_', 'Early-onset_myopathy_with_fatal_cardiomyopathy', 'Elevated_circulating_creatine_kinase_concentration', 'LHB-related_condition', 'FPGT-TNNI3K-related_condition', 'Factor_V_', 'Adiponectin_deficiency', 'PLCG1-related_condition', 'Hypocholesterolemia', 'SOX18-related_condition', 'Ebstein_anomaly', 'VKORC1-related_condition', 'Aneurysm', 'Thrombophilia_3_due_to_protein_C_deficiency', 'Mandibuloacral_dysplasia_with_type_B_lipodystrophy', 'Hypertonia', 'Familial_isolated_congenital_asplenia', 'Hyperapobetalipoproteinemia', 'NFATC1-related_condition', 'MESP1-related_condition', 'TMEM65-related_condition', 'Congenital_heart_defects_and_skeletal_malformations_syndrome', 'ARMC2-related_condition', 'OBESITY_(BMIQ9)', 'CACNA1D-related_condition', 'AMPD3-related_condition', 'Cowden_syndrome', 'GNB5-related_disorder', 'Qualitative_or_quantitative_defects_of_beta-myosin_heavy_chain_(MYH7)', 'CRP-related_condition', 'SERPIND1-related_condition', 'APOE-related_condition', 'Cerebral_atrophy', 'Third_degree_atrioventricular_block', 'NDUFA4-related_condition', 'RYR2-related_condition', 'ITGA3-related_condition', 'AKAP10-related_condition', 'GALNT11-related_condition', 'PAK2-related_condition', 'Premature_ventricular_contraction', 'APOLIPOPROTEIN_C-II_(BARI)', 'Sengers_syndrome', 'PAK1-related_condition', 'Pulmonic_stenosis', 'Paroxysmal_atrial_fibrillation', 'EPO-related_condition', 'LDLRAP1-related_condition', 'Dilated_cardiomyopathy_1C', 'Myofibrillar_Myopathy', 'Chronic_kidney_disease', 'RAP1B-related_condition', 'TNNI3K-related_condition', 'Carney_complex_-_trismus_-_pseudocamptodactyly_syndrome', 'IL1B-related_condition', 'congenital_heart_defects', 'Brugada_syndrome', 'EPHB4-related_disorders', 'Deep_venous_thrombosis', 'Intellectual_developmental_disorder_with_cardiac_defects_', 'ZFHX3-related_condition', 'ZMPSTE24-Related_Disorders', 'Loeys-Dietz_syndrome', 'Sudden_cardiac_failure', 'KCNJ6-related_condition', 'SLC39A8-related_condition', 'PRKD2-related_condition', 'CACNA1D-related_disorder', 'TNNI3-Related_Disorders', 'ITGA6-related_condition', 'FSCN2-related_condition', 'Primary_pulmonary_hypoplasia', 'Danon_disease', 'Apolipoprotein_A-II_deficiency', 'Familial_hypertrophic_cardiomyopathy', 'TPM2-related_condition', 'Cardiac-urogenital_syndrome', 'Low_renin', 'CAP-congenital_myopathy_with_arthrogryposis_multiplex_congenita_without_heart_involvement', 'Mitochondrial_complex_IV_deficiency', 'Coronary_artery_atherosclerosis', 'Multicentric_Osteolysis-Nodulosis-Arthropathy_(MONA)_Spectrum_Disorders', 'Pulmonary_arterial_hypertension_associated_with_another_disease', 'NLRP3-related_condition', 'Dilated_cardiomyopathy_1NN', 'Right_aortic_arch', 'Fetal_Cardiomyopathy', 'PDLIM3-related_condition', 'Left-right_axis_malformations', 'Familial_hypobetalipoproteinemia', 'APOLIPOPROTEIN_C-II_(WAKAYAMA)', 'Glycogen_storage_disease', 'ATP2A2-related_condition', 'KIF6-related_condition', 'DOCK2-related_condition', 'LMNA-associated_condition', 'Cardiomyopathy-hypotonia-lactic_acidosis_syndrome', 'Pseudohypoaldosteronism_type_2E', 'Hereditary_hemochromatosis_type', 'SIM1-associated_metabolic_syndrome', 'RRAS2-related_condition', 'CD40LG-related_condition', 'ITGA2B-related_condition', 'COPD', 'X-linked_Emery-Dreifuss_muscular_dystrophy', 'TBX3-related_condition', 'CYP11B2-related_disorder', 'KCNT2-related_condition', 'Dilated_cardiomyopathy_2A', 'DNAJB2-related_condition', 'Scimitar_syndrome', 'Low_density_lipoprotein_cholesterol_level_quantitative_trait_locus', 'Cardiac_arrest', 'LIPE-related_familial_partial_lipodystrophy', 'FGF16-related_condition', 'JAM2-related_condition', 'FATTY_ACID-BINDING_PROTEIN', 'Dilated_cardiomyopathy_1DD', 'MYBPC3-related_cardiomyopathies', 'desflurane_response_-_Toxicity', 'Neurofibromatosis-Noonan_syndrome', 'Periventricular_nodular_heterotopia', 'SLC25A6-related_condition', 'Thromboembolism', 'Thrombotic_stroke', 'unspecified_heart_condition', 'Hb_SS_disease', 'MT-CYB_associated_Exercise_intolerance', 'Pick_disease', 'Diastolic_dysfunction', 'CNNM2-related_condition', 'Nestor-Guillermo_progeria_syndrome', 'AGBL1-related_condition', 'Carnitine_palmitoyltransferase_II_deficiency', '_anterior_h', 'MT-TK-related_mitochondrial_disorder', 'Hutchinson-Gilford_progeria_syndrome', 'Very_long_chain_fatty_acid_accumulation', 'Primary_familial_hypertrophic_cardiomyopathy', 'CNTN3-related_condition', 'ADGRG2-related_condition', 'STXBP5L-related_condition', 'Loss_of_consciousness', 'Atrial_standstill', 'VEGFA-related_condition', 'PKP2-related_condition', 'APOLIPOPROTEIN_A-IV_POLYMORPHISM', 'TRPM4-Related_Disorders', 'Coronary_heart_disease', 'Diaphragmatic_hernia', 'Rhd', 'CPB2-related_condition', 'LMX1B-related_condition', 'Dehydrated_hereditary_stomatocytosis', 'SLC25A1-related_condition', 'SMAD2-congenital_heart_disease_', 'NDUFA12-related_condition', 'TNNT2-Related_Cardiomyopathy', 'Adrenal_pheochromocytoma', 'SMAD2-related_cardiac_disorders', 'Periventricular_nodular_heterotopia_', 'PTPN11_Related_Disorders', 'PPM1K-related_condition', 'Ventriculomegaly', 'TGFBR2-related_condition', 'Familial_isolated_arrhythmogenic_right_ventricular_dysplasia', 'Ritscher-Schinzel_syndrome', 'Alport_syndrome_3b', 'Immunodeficiency_93_and_hypertrophic_cardiomyopathy', 'SLC4A7-related_condition', 'Myopathy_caused_by_variation_in_FKTN', 'Myokymia_1_with_hypomagnesemia', 'Gaucher_disease_type_III', 'Metabolic_syndrome_X', 'Familial_apolipoprotein_C-II_deficiency', 'ACE-related_condition', 'Carnitine_palmitoyl_transferase_II_deficiency', 'Acrocapitofemoral_dysplasia', 'Intrinsic_cardiomyopathy', 'Impaired_thromboxane_A2_agonist-induced_platelet_aggregation', 'Factor_V_deficiency', 'NKX2-5-related_condition', 'ACTN2-related_condition', 'NDUFS8-related_condition', 'ENTPD1-related_condition', 'Turner_syndrome', 'Coronary_artery_disorder', 'THBS1-related_condition', 'ADIPOQ-related_condition', 'DCHS1-related_disorder', 'Congenital_titinopathy', 'ACTN2-related_disorders', 'Steel_syndrome', 'Abnormality_of_the_face', 'EMILIN1-related_condition', 'Ulnar-mammary_syndrome', 'Cardiac_arrhythmia', 'Congenital_heart_defects_', 'Ventricular_tachycardia', 'Microangiopathy_', 'Doxorubicin_response', 'GP1BA-related_condition', 'LMNA-Related_Disorders', 'Congenital_disorder_of_connective_tissue', 'Hypertensive_disorder', 'Short_sleep', 'FLNB-Related_Spectrum_Disorders', 'Arrhythmogenic_cardiomyopathy_with_variable_ectodermal_abnormalities', 'Paroxysmal_familial_ventricular_fibrillation', 'RASopathy', 'ANK2-related_condition', 'LZTS1-related_condition', 'CACNB2-related_condition', 'Pulmonary_embolism', 'Abnormal_circulating_thyroid_hormone_concentration', 'PITX2-related_condition', 'Hypertrophic_cardiomyopathy', 'KCNQ1OT1-related_condition', 'Hepatorenocardiac_degenerative_fibrosis', 'Familial_isolated_dilated_cardiomyopathy', 'Alagille_syndrome_due_to_a_JAG1_point_mutation', 'Amyloidosis', 'Effort-induced_polymorphic_ventricular_tachycardia', 'Iron_overload', 'RhD_category_D-VII', 'Hyperlipoproteinemia', 'Hermansky-Pudlak_syndrome_with_pulmonary_fibrosis', 'Glycogen_storage_disease_II', 'ACTA2-related_condition', 'APOB-related_disorder', 'Lactic_acidosis', 'MMP13-related_condition', 'VLDLR-related_condition', 'Qualitative_or_quantitative_defects_of_perlecan', 'Hereditary_antithrombin_deficiency', 'neonatal_lactic_acidosis', 'CORIN-related_condition', 'Li-Campeau_syndrome', 'CASQ2-related_condition', 'Thrombocytosis', 'Arterial_calcification', 'TPM3-related_condition', 'Decreased_activity_of_mitochondrial_ATP_synthase_complex', 'TRIM2-related_condition', 'SEMA3D-related_condition', 'FGF12-related_condition', 'PTGIS-related_condition', 'SCN4A-Related_Disorders', 'Coronary_sclerosis', 'Glycogen_storage_disease_IXa1', 'PDK3-related_condition', 'Pseudohypoaldosteronism_type_2D', 'PPARG-Related_Disorders', 'TNNT1-related_condition', 'PDGFB-related_condition', 'SMIM1-related_condition', 'Beta-thalassemia_major', 'Periodontitis', 'Encephalocardiomyopathy', 'UCP2-related_condition', 'FIBRINOGEN', 'Glycogen_storage_disease_IXd', 'MTHFR-related_condition', 'Periventricular_heterotopia_with_microcephaly', 'GALNT2-related_condition', 'Neurodevelopmental-craniofacial_syndrome_with_variable_renal_and_cardiac_abnormalities', 'KCNJ12-related_condition', 'Long_COVID-19', 'Lipoprotein(a)_deficiency', 'Ear_malformation', '_hypomagnesemia', 'Dilated_cardiomyopathy_1V', 'NARP_syndrome', 'BIN1-related_condition', 'Dilated_cardiomyopathy_1E', 'Familial_partial_lipodystrophy', 'MVD-related_condition', 'LDHD-related_condition', 'Hyperalphalipoproteinemia', 'Protein_S_deficiency_disease', 'ABCA1-related_dyslipidemia', 'SLC22A5-related_condition', 'ADAM10-related_condition', 'Neurodevelopmental_disorder_with_cardiomyopathy', 'Susceptibility_to_coronavirus_disease_(COVID-19)_severity_', 'Orthostatic_hypotension', 'Premature_coronary_artery_atherosclerosis', 'Parkes_Weber_syndrome', 'Hypercholesterolemia', '_diabetes', 'Coenzyme_Q10_deficiency', 'DSC2-related_condition', 'Exercise_intolerance', 'MYH6-Related_Disorders', 'AMED_syndrome', 'KCNN2-related_condition', 'SELECTIN_P_POLYMORPHISM', 'Tuberous_sclerosis', 'Dilated_cardiomyopathy-hypergonadotropic_hypogonadism_syndrome', 'LPIN1-related_condition', 'RASA1-related_condition', 'Alex', 'Bicuspid_aortic_valve', 'PTPN11-related_disorder', 'ANEMIA', 'Retinal_arterial_macroaneurysm_with_supravascular_pulmonic_stenosis', 'TNNI3-related_condition', '_skeletal_defects', 'Susceptibility_to_severe_COVID-19', 'CD40-related_condition', 'MIR145-related_multisystemic_smooth_muscle_dysfunction', 'TJP2-related_condition', 'Marfanoid_habitus_', 'Arrhythmogenic_cardiomyopathy_with_wooly_hair_and_keratoderma', 'NDUFA1-related_condition', '_disease_type_2B', 'TBX1-related_condition', 'EDEM3-related_condition', 'Tobacco_use_disorder', 'RGS2-related_condition', 'Vasculitis', 'Dilated_cardiomyopathy_1CC', 'MYL2-related_condition', 'Ezetimibe_response', 'DLL4-related_condition', 'Hereditary_Sideroblastic_Anemia_with_Myopathy_and_Lactic_Acidosis', 'Familial_digital_arthropathy-brachydactyly', 'Mitochondrial_hypertrophic_cardiomyopathy_with_lactic_acidosis_due_to_MTO1_deficiency', 'EMG_abnormality', 'ABL1-related_congenital_heart_defects_', 'DLG4-related_synaptopathy', 'ABCC8-related_condition', 'ZMPSTE24-related_condition', '_autonomic_neuropathy_type', 'TLR4-related_condition', 'Qualitative_or_quantitative_defects_of_alpha-sarcoglycan', 'Dilated_cardiomyopathy_1BB', 'ADD3-related_condition', 'Atrial_flutter', 'Mitochondrial_disease', 'ITGAV-related_condition', 'Portal_hypertension', 'ATP1A1-related_condition', 'TPI1-related_condition', 'Gnb5-related_intellectual_disability-cardiac_arrhythmia_syndrome', 'GUCY1A1-related_condition', 'NTN1-related_condition', 'SCN10A-related_condition', 'ALBUMIN_CASERTA', 'Bardet-Biedl_syndrome', 'NDUFAF4-related_condition', '_Ehlers-Danlos_syndrome', 'LDHA-related_condition', 'CCS-related_condition', 'Autosomal_recessive_limb-girdle_muscular_dystrophy_type_2I', 'Atorvastatin_response', 'FXYD6-FXYD2-related_condition', 'Jeune_thoracic_dystrophy', '/or_familial_pulmonary_arterial_hypertension', 'Myocarditis', 'Combined_osteogenesis_imperfecta_and_Ehlers-Danlos_syndrome', 'FNDC3A-related_condition', 'PAH-related_condition', 'TGFB3-related_connective_tissue_disorders', 'Left_ventricular_noncompaction_cardiomyopathy', 'PIEZO1-related_condition', 'Congenital_Indifference_to_Pain', 'Lysosomal_acid_lipase_deficiency', 'Impaired_ADP-induced_platelet_aggregation', 'CPAMD8-related_condition', 'Carotid_intimal_medial_thickness', 'MYH11-related_condition', 'Dilated_Cardiomyopathy', 'TGFB1-related_condition', 'MYH7B-related_condition', 'Congenital_diaphragmatic_hernia', 'Meretoja_syndrome', 'NDUFAF8-related_condition', 'LDHB-related_condition', 'LRP8-related_condition', 'Pulmonary_venoocclusive_disease', 'TRIM55-related_condition', 'Thrombophilia_due_to_protein_S_deficiency', 'Venous_thromboembolism', 'Combined_immunodeficiency_and_megaloblastic_anemia_with_or_without_hyperhomocysteinemia', 'Protein_Z_deficiency', 'NDUFAF6-related_condition', 'CAV1-related_condition', 'Dilated_cardiomyopathy_1J', 'Fatal_infantile_mitochondrial_cardiomyopathy', 'CARNEY_COMPLEX', 'PCSK9-Related_Disorders', 'Wolfram-like_syndrome', 'Hydrocephalus_due_to_aqueductal_stenosis', 'Neurooculocardiogenitourinary_syndrome', 'ANGPTL6-related_condition', 'Hypoplastic_aortic_arch', 'Marfan_Syndrome/Loeys-Dietz_Syndrome/Familial_Thoracic_Aortic_Aneurysms_', 'HMGCR-related_condition', 'Azorean_disease', 'Mild_liver_congestion', 'Gastrointestinal_defect_and_immunodeficiency_syndrome', 'Arrhythmogenic_right_ventricular_dysplasia', 'COX4I1-related_condition', 'Liddle_syndrome', 'Desmin-related_myofibrillar_myopathy', 'Tricuspid_atresia', 'HSPB8-related_condition', 'Hereditary_arterial_and_articular_multiple_calcification_syndrome', 'AGTR2-related_condition', 'Heart', 'Leptin_dysfunction', 'ACTN1-related_condition', 'Behcet_disease', 'Ehlers-Danlos_syndrome_due_to_tenascin-X_deficiency', 'Cerebral_arteriovenous_malformation', 'ITGA7-related_condition', 'Apolipoproteins_a-i_', 'Abnormality_of_coagulation', 'Familial_thoracic_aortic_aneurysm_', 'Left_ventricular_noncompaction', 'FLNA-related_periventricular_nodular_heterotopia', 'PRRC2A-related_condition', 'Dilated_left_ventricle', 'PPP1CB-related_condition', 'Autosomal_recessive_Kenny-Caffey_syndrome', '_obesity', 'EEM_syndrome', 'Dilated_cardiomyopathy_1X', 'KCNT1-related_channelopathy', 'TGFBR3-related_condition', 'Glycogen_storage_disease_type_II', 'Dilated_cardiomyopathy_1A', 'Thrombophilia_due_to_activated_protein_C_resistance', 'APOB-related_condition', 'PANX1-related_condition', 'APOLIPOPROTEIN_C-II_(TORONTO)', 'ADan_amyloidosis', 'MTR-related_condition', 'ITGA9-related_condition', 'ADRB2-related_condition', 'Glycogen_storage_disease_due_to_glycogen_branching_enzyme_deficiency', 'DMD-related_condition', 'FLNA_related_disorders', 'TPM2-related_myopathy', 'UCP3-related_condition', 'SCN1A-related_channelopathy', 'Diabetes', 'Cyanosis', 'Highly_elevated_creatine_kinase', 'Retinitis_pigmentosa_with_or_without_situs_inversus', 'TIMP3-related_condition', 'Asphyxiating_thoracic_dystrophy', 'Hereditary_hemochromatosis', '_great_vessels', 'Pulmonary_arteriovenous_malformation', 'MYH6-related_cardiac_defects', 'APOC2-related_condition', 'CDH2-related_condition', 'APOLIPOPROTEIN_A-I_(MUNSTER3C)', 'FLNA_-_related_disorder', 'LRPPRC-related_condition', 'Autosomal_recessive_limb-girdle_muscular_dystrophy_type_2C', 'FIBRINOGEN_MUNICH', 'Hemochromatosis', 'ALPK3-related_disorder', 'LEMD2-related_condition', 'TBX5-related_condition', 'Apolipoprotein_c-III_deficiency', 'Warfarin_response', 'CAVIN4-related_condition', 'Familial_amyloid_neuropathy', 'Pulmonary_artery_atresia', 'Dilated_cardiomyopathy_1Y', 'LIPI-related_condition', 'TGFBR1-related_condition', 'Hypomagnesemia', 'FOXF1-related_condition', 'SCN5A-related_conditions', 'Sick_sinus_syndrome', 'TTN-related_disease', 'Progeroid_and_marfanoid_aspect-lipodystrophy_syndrome', 'Fetal_', 'Hyper', '_Lactic_Acidosis', 'Wolff-Parkinson-White_pattern', 'Dilated_cardiomyopathy_1R', 'MMP9-related_condition', 'SLC25A5-related_condition', 'PLA2G2A-related_condition', 'Congenital_heart_disease', 'F2R-related_condition', 'Dermatofibrosis_lenticularis_disseminata', 'Glycogen_storage_disease_IV', 'LCAT_deficiency', 'RYR2-related_disorder', 'COX7B-related_condition', 'MYLK2-related_condition', 'Abnormality_of_the_skin', 'FBN1-related_disease', 'RAMP2-related_condition', 'MYLK3-associated_cardiomyopathy', 'ABCB1-related_condition', 'SH2B3-related_condition', 'Glycogen_storage_disease_IIIb', 'Insulin-resistant_diabetes_mellitus', 'Congestive_heart_failure_', 'TRDN-related_condition', 'Jervell_', 'TRPC3-related_condition', 'Duchenne_and_Becker_muscular_dystrophy', 'Abnormal_renal_pelvis_morphology', 'PNPLA2-related_condition', 'Feingold_syndrome_type', 'FKRP-related_condition', 'Myopathy_caused_by_variation_in_GMPPB', '_distinctive_facial_features_with_cardiac_defects', '_with_or_without_vertebral_or_cardiac_anomalies', 'PRKAR1A-related_condition', 'NOS3-related_condition', 'SFXN4-related_condition', 'Juvenile_polyposis/hereditary_hemorrhagic_telangiectasia_syndrome', 'KCNN3-related_condition', 'EMD-related_condition', 'Structural_heart_defects_', 'DYRK1B-related_condition', 'Torsades_de_pointes', 'EMILIN-1-related_connective_tissue_disease', 'CAMK2A-related_condition', 'TMEM70-related_condition', 'KLHL40-related_condition', 'AARS2-Related_Disorders', 'Testicular_anomalies_with_or_without_congenital_heart_disease', 'PCSK9-related_condition', 'ANKRD1-related_condition', 'ADIPOR2-related_condition', 'VCP-related_multisystem_proteinopathy', 'MED13-related_condition', 'Familial_type_5_hyperlipoproteinemia', 'Friedreich_ataxia', 'EFHC1-related_condition', 'SLC2A10-related_condition', 'PLN-related_condition', 'TBX18-related_condition', 'Glycogen_storage_disease_IXb', 'Channelopathy', 'SLC17A9-related_condition', 'GJA3-related_condition', 'DNAJB13-related_condition', 'HSD11B2-related_condition', '_common_atrioventricular_junction', 'Dilated_cardiomyopathy_1EE', 'Brugada_syndrome_(shorter-than-normal_QT_interval)', 'KCNQ1-Related_Disorders', 'Pulmonary_atresia_with_intact_ventricular_septum', 'ITGA2B-Related_Disorders', 'Neurocardiofaciodigital_syndrome', 'KCNH2-related_disorders', 'Pancreatic_hypoplasia-diabetes-congenital_heart_disease_syndrome', 'Aldosterone-producing_adrenal_adenoma', 'Long_qt_syndrome', 'Hypercoagulability_syndrome_due_to_glycosylphosphatidylinositol_deficiency', 'SCN5A-Related_Disorders', 'Classic_homocystinuria', 'ENPP1-Related_Disorders', 'Hypoprebetalipoproteinemia', 'PRKAG3-related_condition', 'H', 'POPDC3-related_condition', 'MMP2-related_condition', 'Fetal_growth_restriction', 'Intellectual_developmental_disorder_with_cardiac_defects_and_dysmorphic_facies', 'Heart_', 'Intestinal_hypomagnesemia', 'Primary_CD59_deficiency', 'MYBPC2-related_condition', 'Susceptibility_to_severe_coronavirus_disease_(COVID-19)_due_to_high_levels_of_fibrinogen_and_C-reactive_protein', 'Mulibrey_nanism_syndrome', 'CELA2A-related_condition', 'Indifference_to_pain', '_with_or_without_cardiac_or_endocrine_anomalies', 'Cohen_syndrome', 'NODAL-related_condition', 'SMARCA4-related_BAFopathy', 'Atrial_septal_defect', 'M', 'Left_ventricular_hypertrophy', 'CTNNA1-related_condition', 'Preeclampsia', 'MEGF8-related_Carpenter_syndrome', 'Potocki-Lupski_syndrome', 'DLD-related_condition', 'Mandibuloacral_dysplasia_with_type_A_lipodystrophy', 'NOTCH1-related_condition', 'IGFBP7-related_condition', 'Brachydactyly-arterial_hypertension_syndrome', 'SOX6-related_condition', 'TBXA2R-related_condition', 'Thrombophilia_due_to_protein_C_deficiency', 'Vascular_dementia', 'NPR3-related_condition', 'Emery-Dreifuss_muscular_dystrophy', 'Chronic_atrial_', 'NEDD4L-related_condition', 'ERG2-related_disorders', 'Myopathy_with_abnormal_lipid_metabolism', 'HABP2-related_condition', 'NDUFS6-related_condition', 's', 'APOLIPOPROTEIN_C-II_(AUCKLAND)', 'LIPOPROTEIN_LIPASE_(OLBIA)', 'Diabetes_mellitus', 'Familial_cardiofaciocutaneous_syndrome', 'MEF2D-related_condition', 'Familial_pulmonary_capillary_hemangiomatosis', 'MYBPC1-related_condition', 'Syncope', 'Autosomal_dominant_polycystic_kidney_disease', 'Arterial_tortuosity', 'Progressive_pulmonary_failure', 'TRAF7-associated_heart_defect-digital_anomalies-facial_dysmorphism-motor_', 'EFNB2-related_condition', 'Choanal_atresia-hearing_loss-cardiac_defects-craniofacial_dysmorphism_syndrome', 'DAB1-related_condition', 'Progressive_familial_heart_block', 'TGFB2-related_condition', 'CHRM3-related_condition', 'Cystic_fibrosis', 'Singleton-Merten_syndrome', 'SMAD6-related_condition', 'ITPR3-related_condition', 'POLYMICROGYRIA_WITHOUT_VASCULAR-TYPE_EHLERS-DANLOS_SYNDROME', 'Double_outlet_right_ventricle', 'Familial_hyperaldosteronism_type_III', 'Autoimmune_connective_tissue_disease_and_vasculitis', 'Vitamin_K-Dependent_Clotting_Factors', 'CACNA1G-related_condition', 'PVR-related_condition', 'Decreased_circulating_carnitine_concentration', 'Congenitally_corrected_transposition_of_the_great_arteries', 'H1-4-related_condition', 'SCN1B-related_condition', 'TTN-Related_Disorders', 'Meester-Loeys_syndrome', 'Coenzyme_q10_deficiency', 'ARSA-related_condition', 'Mitochondrial_DNA_depletion_syndrome_12A_(cardiomyopathic_type)', 'CP-related_condition', 'Two-raphe_bicuspid_aortic_valve', 'Primary_hypomagnesemia', 'Cholesteryl_ester_storage_disease', 'Uruguay_Faciocardiomusculoskeletal_syndrome', 'PDE3A-related_condition', 'Familial_dilated_cardiomyopathy_and_peripheral_neuropathy', 'Lessel-kubisch_syndrome', 'ALPK3-related_condition', 'CAP2-associated_dilated_cardiomyopathy', 'Infantile_hypertrophic_cardiomyopathy_due_to_MRPL44_deficiency', 'TRAF7-associated_heart_defect-digital_anomalies-facial_dysmorphism-motor_and_speech_delay_syndrome', 'MYBPC3-related_disease', 'Serum_amyloid_a_variant', 'ANKRD1-related_dilated_cardiomyopathy', 'Glomuvenous_malformation', 'Familial_High_Density_Lipoprotein_Deficiency', 'Statins', 'Glycogen_storage_disease_IXc', 'PTPN11-related_condition', 'Deafness-encephaloneuropathy-obesity-valvulopathy_syndrome', 'Abnormal_aortic_valve_physiology', 'CACNA2D1-related_condition', 'APOLIPOPROTEIN_A-I_(GIESSEN)', 'LAMA5-related_condition', 'ACADVL-related_condition', 'CACNA2D3-related_condition', 'Severe_hypotonia-psychomotor_developmental_delay-strabismus-cardiac_septal_defect_syndrome', 'Congestive_heart_failure', 'BVES-related_condition', 'VACTERL_association', 'MMP14-related_condition', 'DCHS2-related_condition', 'COX15-related_condition', 'NPC1L1-related_condition', 'Myhre_syndrome', 'Patent_ductus_arteriosus', 'Carnitine_acylcarnitine_translocase_deficiency', 'Hereditary_hemorrhagic_telangiectasia', 'PTGS1-related_condition', 'Multisystem_inflammatory_syndrome_in_children', 'Recurrent_metabolic_encephalomyopathic_crises-rhabdomyolysis-cardiac_arrhythmia-intellectual_disability_syndrome', 'DiGeorge_syndrome', 'Ptosis', 'Neonatal_encephalomyopathy-cardiomyopathy-respiratory_distress_syndrome', 'GAPVD1-related_condition', '_myopathy_syndrome', 'COQ5-related_condition', 'FBN3-related_condition', 'Abnormality_of_acid-base_homeostasis', 'PPARGC1B_polymorphism', 'KCNJ11-related_condition', 'ANK1-related_condition', 'Thiamine_Metabolism_Dysfunction_Syndrome', 'Noonan_syndrome-like_disorder_with_loose_anagen_hair', 'DMPK-related_condition', 'ANGPT2-related_condition', 'GUF1-related_condition', 'HSPG2-realted_disorder', 'Duchenne_', 'MYOC-Related_Disorders', 'Idiopathic_pulmonary_arterial_hypertension', 'Sitosterolemia', 'NOTCH1-Related_Disorders', 'KCNQ5-related_condition', 'Ventriculomegaly_', 'WNT2B-related_condition', 'Congenital_long_QT_syndrome', 'CAMK2B-related_condition', 'SARS2-related_condition', 'Trichomegaly', 'C_syndrome', 'PPARG-related_familial_partial_lipodystrophy', 'Costello_syndrome', 'ABCA1-Related_Disorders', 'AGTR1-related_condition', 'Thrombophilia_caused_by_F2_prothrombin_deficiency', 'Progressive_external_ophthalmoplegia_with_mitochondrial_dna_deletions', 'Sinoatrial_node_disorder', 'SNTA1-related_condition', 'FGF13-related_condition', 'Systolic_heart_failure', 'Cardioencephalomyopathy', 'JPH2-related_condition', 'Abnormal_platelet_function', 'HEMOGLOBIN_CAPA', 'MEGF8-related_condition', 'Dilated_cardiomyopathy_1G', 'Pulmonary_arterial_hypertension_related_to_hereditary_hemorrhagic_telangiectasia', 'APOC3-related_condition', 'Primary_pulmonary_hypertension', 'Non-immune_hydrops_fetalis', 'NDUFAF3-related_condition', 'Muscular_dystrophy', 'ABL1-related_congenital_heart_defects_and_skeletal_malformations_syndrome', 'NDUFV3-related_condition', 'Atrioventricular_septal_defect', 'Tuberous_sclerosis_', 'MEF2A-related_condition', 'NDRG4-related_condition', 'Naxos_disease', 'LTBP2-related_condition', 'SCNN1B-related_condition', 'Familial_aortic_aneurysms', 'Aortic_dissection', 'RREB1-associated_Noonan-like_syndrome', 'MYH7-Related_Disorders', 'Dilated_cardiomyopathy_1O', 'APOLIPOPROTEIN_A-I_(MUNSTER3B)', 'TNNT2_-related_cardiomyopathies', 'Decreased_activity_of_mitochondrial_complex_IV', 'Polyps', 'Vascular_endothelial_growth_factor_(VEGF)_inhibitor_response', 'WNT11-related_condition', 'RNF213-related_condition', 'MYH8-related_condition', 'Dilated_cardiomyopathy_2B', 'GLRX5-related_condition', 'APOB-Related_Disorders', 'MYOPATHY', 'KCNMB1-related_condition', 'APOLIPOPROTEIN_A-I_(MUNSTER4)', 'Warfarin_sensitivity', 'KLHL7-related_condition', 'TTR-related_condition', 'Caveolinopathy', 'MMP3-related_condition', 'Hypercalcemia', 'Arrhythmogenic_ventricular_cardiomyopathy', 'Spasticity', 'UNC45B-related_condition', 'COQ9-related_condition', 'Abnormal_thrombosis', 'FOXO1-related_condition', 'Familial_restrictive_cardiomyopathy', 'TRPC6-related_condition', '_hyper', 'CACNA2D2-related_condition', 'EDN3-related_condition', 'RAF1-related_disorders', 'PDE1C-related_condition', 'Isolated_congenital_digital_clubbing', 'Risk_of_requirement_of_invasive_mechanical_ventilation_in_patients_with_severe_COVID-19', 'SEMA3C-related_condition', 'Congenital_heart_defects_and_ectodermal_dysplasia', 'LEOPARD_syndrome', 'Multicentric_osteolysis_nodulosis_arthropathy_spectrum', 'ADRA2B-Related_Disorder', 'FBN2-related_condition', 'MOCS2-related_condition', 'MYH7-related_condition', 'Atypical_coarctation_of_aorta', 'Familial_type_3_hyperlipoproteinemia', 'ABCD3-related_condition', 'ADAMTS19-associated_congenital_heartdefect', 'Ehlers-danlos_syndrome', 'LMF1-related_condition', 'POC5-related_condition', 'Atherosclerosis', 'Sudden_unexplained_death_in_childhood', 'Velocardiofacial_syndrome', 'Lpl-arita', 'Nicotine_dependence', 'MEIS1-related_condition', 'JAG1-related_condition', 'APOLIPOPROTEIN_A-II_DEFICIENCY', 'SGCD-related_condition', 'SCN10A-Related_Disorder', 'Enzyme_activity_finding', 'Pyloric_stenosis', 'ICAM1-related_condition', 'Overhydrated_hereditary_stomatocytosis', 'Cystinosis', '_hypertrophic_cardiomyopathy', 'Mitochondrial_DNA_depletion_syndrome_14_(cardioencephalomyopathic_type)', 'Bardet-biedl_syndrome_1/10', 'Dilated_cardiomyopathy_1AA', 'Connective_tissue_dysplasia', 'BAG3-related_condition', 'DNAJC19-related_condition', 'AKT1-related_condition', 'Deficiency_of_malonyl-CoA_decarboxylase', 'PRKCA-related_condition', 'FCN3-related_condition', 'Simpson-Golabi-Behmel_syndrome_type', 'Camptodactyly-arthropathy-coxa_vara-pericarditis_syndrome', 'Carney-Stratakis_syndrome', 'AKT2-related_condition', 'COQ6-related_condition', 'ATF6-related_condition', 'KCNJ2-related_condition', 'IPO8-related_aortopathy', '3M_syndrome', 'LPA-related_condition', 'Angiokeratoma_corporis_diffusum_with_arteriovenous_fistulas', 'Long_chain_acyl-CoA_dehydrogenase_deficiency', 'Noonan-like_syndrome', 'TGFB3-related_condition', '_flapping', 'DNAJB6-related_condition', 'Cardio-cutaneous_syndrome', 'MIF-related_condition', 'ADIPOR1-related_condition', 'FG_syndrome', 'X-linked_intellectual_disability-cardiomegaly-congestive_heart_failure_syndrome', 'Familial_aortopathy', 'Aural_atresia', 'NR2F2_associated_disorders', 'TREX1-related_condition', 'Idiopathic_and/or_familial_pulmonary_arterial_hypertension', 'MFAP5-related_condition', 'Abnormal_platelet_aggregation', 'Malformation_of_the_heart_and_great_vessels', 'C1QTNF5-related_condition', 'PRKG1-related_condition', 'ANGPT1-related_condition', 'Heart_block', 'Autoimmune_connective_tissue_disease_', 'FGFRL1-related_condition', 'Mitochondrial_complex_5_(ATP_synthase)_deficiency', 'Dilated_cardiomyopathy_1W', 'Predisposition_to_dissection', 'B', 'Haemorrhagic_telangiectasia', 'CD36-Related_Disorders', 'COL4A1-related_condition', 'GPIHBP1-related_condition', 'GCH1-related_condition', 'WNK4-related_condition', 'GDF2-related_condition', 'LAMA3-related_condition', 'q22)', 'Hyperproreninemia', 'FLT1-related_condition', 'FHL1-related_condition', 'Aldosterone_to_renin_ratio', 'Vitamin_K-dependent_clotting_factors', 'Dextrocardia', 'GATA6-related_condition', 'NKX2-6-related_condition', 'GIT1-related_condition', 'Systemic_lupus_erythematosus', 'LOW_DENSITY_LIPOPROTEIN_CHOLESTEROL_LEVEL_QUANTITATIVE_TRAIT_LOCUS', 'ZFPM2-related_condition', 'Pulmonary_arterial_hypertension', 'Familial_atrial_myxoma', 'PLATELET_GLYCOPROTEIN_Ib_POLYMORPHISM', 'GNB5-Related_Disorders', 'Atrial_st', 'SORT1-related_condition', 'MIP-related_condition', 'Gaucher_disease-ophthalmoplegia-cardiovascular_calcification_syndrome', 'Dilated_cardiomyopathy_1D', 'KCNMA1-related_condition', 'TARP_syndrome', 'Reynolds_syndrome', 'DSG2-related_condition', 'TCF21-related_condition', 'Restrictive_cardiomyopathy', 'Atopy', 'Pulmonary_arterial_hypertension_associated_with_congenital_heart_disease', 'Congenital_muscular_dystrophy_with_rigid_spine', 'PPARG-related_condition', 'MYBPC3-related_disorder', 'HR-related_condition', 'MTTP-related_condition', 'Hypothyroidism_due_to_TSH_receptor_mutations', 'Isolated_thoracic_aortic_aneurysm', 'HEMOGLOBIN_MITO', '_Noonan-related_syndrome', '_SERCA1_protein_overload', 'GP9-related_condition', 'Susceptibility_to_angioedema_induced_by_ACE_inhibitors', 'Familial_atrial_fibrillation', 'Rubinstein-Taybi_syndrome_due_to_CREBBP_mutations', 'Three_Vessel_Coronary_Disease', 'APEX1-related_condition', 'Cardiomyopathy', 'Cardiac_conduction_defect', 'KCNQ1-related_condition', 'Microvascular_complications_of_diabetes', 'Familial_hyperaldosteronism', '-foot_malformation', 'IRX5-related_condition', 'SERPINE1-related_condition', 'MYL2-related_disease', 'Carnitine_palmitoyl_transferase_1A_deficiency', 'VISS_syndrome', 'Large_vessel_vasculitis', 'GYS2-related_condition', 'ABCG1-related_condition', 'ANE_syndrome', 'UQCRB-related_condition', 'SCN4A-related_condition', 'Drug-_or_toxin-induced_pulmonary_arterial_hypertension', 'COQ4-related_condition', 'Glycemia_variation', 'GRK5-related_condition', 'AMPD1-related_condition', 'PSEN2-related_condition', 'Vici_syndrome', 'Hyperlipidemia', '_diabetes_mellitus', 'Sneddon_syndrome', 'ADRA2A-related_condition', 'Homocystinuria', 'HSPG2-related_condition', 'Ascending_aortic_dissection', 'Familial_hyperaldosteronism_type_II', 'MEF2C-related_condition', 'NEBL-related_condition', 'HR-Related_Disorders', 'ENPP1-related_condition', 'Ventricular_fibrillation', 'ULK4-related_condition', 'ASXL2-related_condition', 'BSND-related_condition', 'CPT1B-related_condition', 'Thoracic_aortic_disease', 'GPR180-related_condition', 'Chromosome_17q11.2_deletion_syndrome', 'Idiopathic_cardiomyopathy', 'CSRP3-related_condition', 'MMP1-related_condition', 'Encephalopathy-hypertrophic_cardiomyopathy-renal_tubular_disease_syndrome', 'CALM2-related_condition', 'Drash_syndrome', 'Heart_defect_-_tongue_hamartoma_-_polysyndactyly_syndrome', 'Thoracic_aortic_aneurysm', 'Primary_dilated_cardiomyopathy', 'MYBPC3-Related_Disorders', 'KCNE3-related_condition', 'PCSK5-related_condition', 'Heart_and_brain_malformation_syndrome', 'ADCK2-related_condition', 'ANGPTL4-related_condition', 'SCN5A-related_disorder', 'Otofaciocervical_syndrome', 'Shprintzen-Goldberg_syndrome', 'Alveolar_capillary_dysplasia_with_pulmonary_venous_misalignment', 'Familial_amyloid_polyneuropathy', 'Concentric_hypertrophic_cardiomyopathy', 'TBX2-related_condition', 'Progeroid_mandibuloacral_dysplasia', 'RAC1-related_condition', 'Congenital_muscular_dystrophy_due_to_LMNA_mutation', 'Autosomal_dominant_MYH7-related_disorder', 'Aborted_sudden_cardiac_death', 'Malformation_of_the_heart_', 'Primary_coenzyme_Q10_deficiency', 'Familial_C', 'NDUFV1-related_condition', 'COX4I2-related_condition', 'NR2F2-related_congenital_heart_defects', 'SDHB-related_condition', 'ADRB1-related_condition', 'Temtamy_syndrome', 'TPM2-Related_Disorders', 'Susceptibility_to_severe_coronavirus_disease_(COVID-19)_due_to_an_impaired_coagulation_process', '_acral', 'Rubinstein-Taybi_syndrome_due_to_EP300_haploinsufficiency', 'WNK1-related_condition', 'Berardinelli-Seip_congenital_lipodystrophy', 'Schimke_immuno-osseous_dysplasia', 'Noonan_syndrome_', 'LRP6-related_condition', 'APOLIPOPROTEIN_A-I_(MILANO)', 'PPP1R13L-associated_cardiac_phenotype', 'Venous_thrombosis', 'Thyrotoxic_periodic_paralysis', 'Pericarditis', 'THYROID_CARCINOMA_WITH_THYROTOXICOSIS', 'Hyperinsulinemia', 'Endocardial_fibroelastosis', 'Meacham_syndrome', 'Cutis_laxa_with_severe_pulmonary', 'Heparin_cofactor_II_deficiency', 'TFAP2B-related_condition', 'Hyperkalemia', 'THRB-related_condition', 'Apolipoprotein_C-III', 'Collapse_(finding)', 'Isolated_Nonsyndromic_Congenital_Heart_Disease', 'Apolipoprotein_A-I_deficiency', 'Noonan_syndrome_and_Noonan-related_syndrome', 'Family_history_of_cardiomyopathy', 'TMEM38B-related_condition', 'FLNC-associated_cardiomyopathy', 'CASQ1-related_condition', 'Dilated_cardiomyopathy_1M', 'Thyroid_hormone_metabolism', 'ACTC1-related_condition', 'Glucocorticoid-remediable_aldosteronism', 'SCN4A-related_disorder', 'GJA5-related_condition', 'WHIM_syndrome', '11q_partial_monosomy_syndrome', 'Congenital_aneurysm_of_ascending_aorta', 'SLC25A4-related_condition', 'CACNA1C-related_disorder', 'Mandibuloacral_dysplasia_progeroid_syndrome', 'CHN2-related_condition', 'Hypokalemic_periodic_paralysis', 'Mortality_risk_in_patients_with_severe_coronavirus_disease_(COVID-19)', 'KRIT1-Related_Disorders', 'OBSCN-related_condition', 'Kabuki_syndrome', 'SLC25A20-related_condition', 'Short_QT_syndrome_type', 'Andersen_Tawil_syndrome', 'Usher_syndrome', 'OBESITY_(BMIQ14)', 'L1_syndrome', 'Falls', 'FXYD2-related_condition', 'Ventriculomegaly_and_arthrogryposis', 'DPP6-related_condition', 'Mitochondrial_trifunctional_protein_deficiency_2_with_myopathy_', 'COL6A2-related_condition', 'Marfan_syndrome', 'FGF10-related_condition', 'CTF1-related_condition', '_Becker_muscular_dystrophy', 'Periventricular_leukomalacia', 'FLNB-related_condition', 'Sepsis', 'UQCRC1-related_condition', 'X-linked_Alport_syndrome', 'NR2F2-related_condition', 'Teebi_hypertelorism_syndrome', 'Laterality_defect_and_complex_congenital_heart_disease', 'SGCA-related_condition', 'Duchenne_muscular_dystrophy', 'sporadic_abdominal_aortic_aneurysm', 'Noonan_syndrome', 'KCNK4-related_condition', 'Alcoholism', 'FBN1-Related_Disorders', 'SCN2B-related_condition', 'Fibromuscular_dysplasia', 'Qualitative_or_quantitative_defects_of_delta-sarcoglycan', 'KCNN4-related_condition', 'PKD2-related_condition', 'Periventricular_nodular_heterotopia_with_syndactyly', 'Right_ventricular_cardiomyopathy', 'BACE2-related_condition', 'Hypokalemia', 'TBXAS1-related_condition', 'Pseudohypoaldosteronism_type_2C', 'Situs_inversus', 'RYR3-related_condition', 'MSX2-related_condition', 'NOL3-related_condition', 'Atrioventricular_septal_defect_', 'PTX3-related_condition', '_megaloblastic_anemia_with_or_without_hyperhomocysteinemia', 'ATP2A1-related_condition', 'Glycogen_storage_disease_IIIc', 'Keutel_syndrome', 'NRP1-related_condition', 'CPT2-related_condition', 'Angiosarcoma', 'WNT5A-related_condition', 'MAP2K1-related_rasopathy-like_syndrome', 'Hyperkalemic_periodic_paralysis', 'FBLN5-related_condition', 'SMAD3-related_condition', 'BMPR2-related_condition', 'Combined_low_LDL_', 'Hydrops-lactic_acidosis-sideroblastic_anemia-multisystemic_failure_syndrome', 'Chronic_atrial_and_intestinal_dysrhythmia', 'Hypoalphalipoproteinemia', 'Ellis-van_Creveld_syndrome', 'HYPERHOMOCYSTEINEMIA', 'Metabolic_syndrome', 'LAMP2-related_condition', 'PURA_Syndrome', 'SUDDEN_INFANT_DEATH_SYNDROME', 'Catecholaminergic_polymorphic_ventricular_tachycardia', 'Sinoatrial_node_dysfunction_', 'NDUFA6-related_condition', 'KCNH1-related_disorder', 'Idiopathic_hypereosinophilic_syndrome', 'Aldosterone-producing_adrenal_cortex_adenoma', 'LMNA-related_condition', 'MYH6-related_condition', 'Kearns-Sayre_syndrome', 'LIMA1-related_condition', 'CFI-related_condition', 'Cardiofacioneurodevelopmental_syndrome', 'ADCY2_related_condition', 'Palpitations', 'Sudden_cardiac_death', 'TMEM43-related_condition', 'HCN1-related_condition', 'Myotonic_dystrophy', 'Dopamine_beta-hydroxylase_polymorphism', 'COG4-related_condition', 'CCDC40-related_condition', 'JPH1-related_condition', 'COX10-related_condition', 'Supravalvar_aortic_stenosis', 'FLNC-related_condition', 'prenatal_LIG4_syndrome_with_aqueductal_stenosis', 'Denticles', 'Mitochondrial_complex_4_deficiency', 'NDUFA2-related_condition', 'Obesity_due_to_CEP19_deficiency', 'ADRA2C-related_condition', 'CELSR2-related_condition', 'Chronic_lung_disease', 'HEMOGLOBIN_RAMPA', 'SREBF2-related_condiiton', 'Hypertrichotic_osteochondrodysplasia_Cantu_type', 'Cardiac_valvular_dysplasia', 'SARS2-associated_condition', 'RSPO2-related_condition', 'Neurodevelopmental_disorder_with_relative_macrocephaly_and_with_or_without_cardiac_or_endocrine_anomalies', 'Abnormal_finger_morphology', 'Alstrom_syndrome', 'OBESITY_(BMIQ17)', 'OSBP-related_condition', 'Neurocirculatory_asthenia', 'Short_QT_Syndrome', 'Clonus', 'Angiotensin_i-converting_enzyme', 'TTN-related_myopathy', 'Fabry_disease', 'ABeta_amyloidosis', 'Cocaine-Related_Disorders', 'TTN-related_condition', 'Increased_mean_platelet_volume', 'SEMA3E-related_condition', 'PDE1A-related_condition', 'Alagille_syndrome_due_to_a_NOTCH2_point_mutation', 'Hyperaldosteronism', 'AIFM1-related_condition', 'Metabolic_crises_with_rhabdomyolysis', 'Hypothyroidism', '_syndrome', 'Ehlers-Danlos_syndrome', 'Preterm_intraventricular_hemorrhage', 'Complex_I_deficiency', 'Qualitative_or_quantitative_defects_of_beta-sarcoglycan', 'TNS1-related_condition', 'DNAH11-related_condition', 'S', 'KCNK5-related_condition', '_Lange-Nielsen_syndrome', 'ABri_amyloidosis', '_C-reactive_protein', 'ATP2B1-related_condition', 'Cardiac_malformation', 'RECLASSIFIED_-_ADRB2_POLYMORPHISM', 'Conotruncal_defect', 'ADA2-related_condition', 'CIDEC-related_familial_partial_lipodystrophy', 'Vascular_Malformations_', 'Vascular_malformation', 'GDF11-related_condition', '_hemochromatosis', 'Short_telomere_length', 'Hydrops_fetalis', 'CAMK2G-related_condition', 'CNTN4-related_condition', 'UQCRQ-related_condition', 'Long_QT_syndrome_2/5', 'Heart_failure', 'C', 'APOA4-related_condition', 'Pulmonary_hypertension', 'Type_2_diabetes_mellitus', 'SIRT3-related_condition', 'TNS2-related_condition', 'Visceral_heterotaxy', 'Ventricular_septal_defect', 'PARAOXONASE_2_POLYMORPHISM', 'DNHD1-related_condition', 'Myopathy_due_to_calsequestrin_and_SERCA1_protein_overload', 'CD93-related_condition', 'GP6-related_condition', 'Aortic_dilatation', 'Long_QT_syndrome', 'Abnormal_brain_morphology', '_immunity_with_systemic_inflammation_', '_complex_III_deficiency', 'AKAP9-related_condition', 'Family_history_of_sudden_cardiac_death', 'ITGA4-related_condition', '_aortic_dissection', 'EPHB4-related_condition', 'SLC16A1-related_condition', 'Hyperlipidemia_due_to_hepatic_triglyceride_lipase_deficiency', 'Stroke_disorder', 'pravastatin_response_-_Efficacy', 'USHER_SYNDROME', 'ABCC9-Related_Disorders', 'HSPB1-Related_Disorder', 'CELSR1-associated_congenital_heartdefects', 'Isolated_h', 'NDUFA10-related_condition', 'PRDM6-related_condition', 'CETP-related_condition', 'Thrombotic_microangiopathy', 'Congenital_heart_defects', 'ERBB4-related_condition', 'HCN2-related_condition', 'Sodium_channelopathy-related_small_fiber_neuropathy', 'MYL3-related_condition', 'ADAMTSL2-related_condition', 'ROBO2-related_condition', 'Mitral_regurgitation', 'P2RY12-related_condition', 'UQCRC2-related_condition', 'KCNQ1-related_epilepsy', 'Supraventricular_tachycardia', 'TRPM7-related_condition', 'Cardiospondylocarpofacial_syndrome', 'SMPD1-related_condition', 'Bardet-biedl_syndrome_2/6', 'RBFOX1-related_condition', 'ARRHYTHMOGENIC_RIGHT_VENTRICULAR_DYSPLASIA', 'Arrhythmogenic_cardiomyopathy', 'NODAL-Related_Disorders', 'HEMOGLOBIN_TAMPA', 'Insulin_resistance', 'Familial_Hypertrophic_Cardiomyopathy_with_Wolff-Parkinson-White_Syndrome', 'Obesity_due_to_SIM1_deficiency', 'TIA1-related_condition', '_complex_congenital_heart_disease', 'Dilated_cardiomyopathy_1Z', 'HEMOGLOBIN_TRENTO', 'ITPKC-related_condition', 'Familial_sick_sinus_syndrome', '_cardiac_abnormalities', 'SREBF2-related_condition', 'IL6R-related_condition', 'KCND3-Related_Disorder', 'Sudden_unexplained_death', 'Acute_aortic_dissection', 'Abnormal_cerebral_morphology', 'TNNC1-related_condition', 'Embryonic_calcium_dysregulation', 'X-linked_intellectual_disability_with_marfanoid_habitus', 'FH-Related_Disorders', 'Rubinstein-Taybi_syndrome', '_beta-blocker_response', 'KCNJ8-related_condition', 'Immunodeficiency_80_with_or_without_congenital_cardiomyopathy', 'Heritable_Thoracic_Aortic_Disease', 'Mitral_valve_prolapse', 'FLNA-related_condition', 'LTBP3-related_condition', 'POMT2-related_condition', 'Familial_dilated_cardiomyopathy_', 'Stage_5_chronic_kidney_disease', 'APOLIPOPROTEIN_A-I_(MARBURG)', 'MFN2-related_condition', 'Acute_coronary_syndrome', 'CHCHD10-related_condition', 'Feingold_syndrome', 'Congenital_', 'CDK13-related_condition', 'CARNITINE_PALMITOYLTRANSFERASE_IA_POLYMORPHISM', 'LARS2-related_condition', 'AHDC1-related_intellectual_disability_-_obstructive_sleep_apnea_-_mild_dysmorphism_syndrome', 'KALRN-related_condition', 'NOS1-related_condition', 'PRKAG2-related_condition', 'FH-related_condition', 'ATR-related_condition', 'Congestive_heart_failure_and_beta-blocker_response', 'Interstitial_cardiac_fibrosis', 'STXBP5-related_condition', 'HEMOGLOBIN_OLEANDER', 'Familial_atrioventricular_septal_defect', 'ADAM17-related_condition', 'Congenital_total_pulmonary_venous_return_anomaly', 'Sudden_cardiac_arrest', 'Chromosome_22q11.2_deletion_syndrome', 'PDA1', 'Dilated_cardiomyopathy_1P', 'Dilated_cardiomyopathy_1KK', 'Timothy_syndrome', 'Dilated_cardiomyopathy_1GG', 'PCM1-related_condition', 'Hypotension', 'HEMOGLOBIN_MAPUTO', 'Decreased_activity_of_mitochondrial_complex_III', '_marfanoid_aspect-lipodystrophy_syndrome', 'SERPINH1-related_condition', 'Hypocalcemia', '_fibrinogen', 'Resistance_to_thyroid_hormone_due_to_a_mutation_in_thyroid_hormone_receptor_beta', 'Interstitial_pulmonary_disease', 'Sinoatrial_node_dysfunction_and_deafness', 'SLC6A2-related_condition', 'FGF23-related_condition', 'Acyl-CoA_dehydrogenase_9_deficiency', 'TBX20-related_condition', 'MTPAP-related_condition', 'Alcohol_dependence', 'Dilated_cardiomyopathy_3B', 'MAP1B-related_condition', 'Hernia', '_vasculitis', 'PHKG2-related_condition', 'ACADS-related_condition', 'Congenital_generalized_lipodystrophy_type', 'CXCL12-related_condition', 'Dilated_cardiomyopathy_1L', 'HMX2-related_condition', 'Channelopathy-associated_congenital_insensitivity_to_pain', 'Noncompaction_cardiomyopathy', 'AARS2-related_condition', 'SOX17-related_condition', 'SCN3A-related_condition', 'Jervell_and_Lange-Nielsen_syndrome', 'GANAB-related_condition', 'ADCY1-related_condition', 'Hemochromatosis_type_2A', 'FLNB-Related_Disorder', 'H_syndrome', 'AV_Block_Third_Degree_Adverse_Event', 'Periventricular_nodular_heterotopia_and_epilepsy', 'Susceptibility_to_severe_coronavirus_disease_(COVID-19)_due_to_high_levels_of_fibrinogen_', 'ATP6AP2-related_condition', 'HBA1-related_condition', 'AKAP14-related_condition', 'High_density_lipoprotein_cholesterol_level_quantitative_trait_locus', 'NFIB-related_condition', 'DLD-Related_Disorders', 'Chromatinopathy', 'Lethal_left_ventricular_non-compaction-seizures-hypotonia-cataract-developmental_delay_syndrome', 'INTERLEUKIN_6_POLYMORPHISM', 'ECE1-related_condition', 'Uric_acid_concentration', 'Aortic_aneurysm', 'FIBRINOGEN-BETA_POLYMORPHISM', 'SOX7-related_condition', 'Morbid_obesity', 'Conotruncal_anomaly_face_syndrome', 'Renal_insufficiency', 'Intellectual_developmental_disorder_with_muscle_tone_abnormalities_and_distal_skeletal_defects', 'Pulmonary_disease', 'EHLERS-DANLOS_SYNDROME', 'SERPINC1-related_condition', 'Duane-radial_ray_syndrome', 'Familial_hypercholesterolemia', 'Hypoplastic_left_heart_syndrome', 'ADCY6-related_condition', 'TANC2-related_condition', 'APOLIPOPROTEIN_C-II_(ST._MICHAEL)', 'RBM10-related_condition', 'Long_QT_syndrome_1/2', 'CD59-related_condition', 'Hypertension', 'HYPOMAGNESEMIA', 'PDE4D-related_condition', 'NDUFAF5-related_condition', 'PLA2G7-related_condition', 'LRP10-related_condition', 'Early-onset_coronary_artery_disease', 'TCF7L2-related_condition', 'Polyarteritis_nodosa', 'Rol', 'Roifman_syndrome', 'CYP2C19:_decreased_function', 'EDNRA-related_condition', 'PPT1-related_condition', 'Brody_myopathy', 'ADARB1-related_condition', 'KCNE5-related_condition', 'HSPG2-Related_Disorders', 'SDHB-Related_Disorders', 'Hypertrophic_osteoarthropathy', 'GPR68-related_condition', 'Persistent_truncus_arteriosus', 'Skeletal_defects', 'CTNND2-related_condition', 'DSP-related_cardiomyopathy', 'SCN3B-related_condition', 'Homozygous_familial_hypercholesterolemia', 'XIRP1-related_condition', 'LDB3-related_condition', 'Short_QT_syndrome', 'MYOC-related_condition', 'Autosomal_dominant_KCNQ1-related_disease', 'APOLIPOPROTEIN_C-II_(PADOVA)', 'Atrioventricular_block', 'FBN1-related_disorder', 'Juvenile_hemochromatosis', 'TTN-Related_disorder', 'Associated_with_severe_COVID-19_disease', 'SREBF1-related_condition', 'Tangier_disease', 'JAG1-related_disorders', 'clopidogrel_response_-_Metabolism/PK', 'DHCR7-related_condition', 'PRDM16-related_congenital_heart_disease', 'LIPOPROTEIN(a)_POLYMORPHISM', 'Ventricular_arrhythmias_due_to_cardiac_ryanodine_receptor_calcium_release_deficiency_syndrome', 'Preeclampsia/eclampsia', 'Everolimus_response', 'SLC12A3-related_condition', 'Lethal_congenital_glycogen_storage_disease_of_heart', 'Congenital_heart_disease_(variable)', 'Hemochromatosis_type', 'Gastrointestinal_defects_', 'Mitochondrial_complex_3_deficiency', 'Dilated_cardiomyopathy_1HH', 'KCNE1-related_condition', 'ACADL-related_condition', 'ZIC3-related_condition', 'Arrhythmogenic_right_ventricular_cardiomyopathy', '_type', 'Patent_foramen_ovale', 'Histiocytoid_cardiomyopathy', 'TAFAZZIN-Related_Disorders', 'Apolipoproteins_a-i_and_c-iii', 'GNAI2-related_condition', 'Tobacco_addiction', 'MYH7-related_disease', 'CAD-related_condition', 'Geleophysic_dysplasia', 'OLR1-related_condition', 'Neonatal_Marfan_syndrome', 'Myopathy', 'Primary_familial_dilated_cardiomyopathy', 'Pheochromocytoma', 'Essential_hypertension', 'AGPAT2-related_condition', 'Stüve-Wiedemann_syndrome', 'TLR2-related_condition', '_spasticity', 'TNXB-related_hypermobile_Ehlers-Danlos_syndrome', 'Dilated_cardiomyopathy_1I', 'LARS1-related_condition', 'NT5E-Related_Disorder', 'GLS-related_condition', 'THBS2-related_condition', 'Dilated_cardiomyopathy_1FF', 'LTBP4-related_condition', 'Acute_myocardial_infarction', 'Pleural_effusion', 'KCNJ11-Related_Disorders', 'SLC25A3-related_condition', 'Myofibrillar_myopathy', 'Dilated_cardiomyopathy_1JJ', 'ABCC8-related_disorders', 'SERPINA1-related_condition', 'SEMA6D-related_condition', 'Alpha-1-antitrypsin_deficiency', 'NDUFV1-Related_Disorders', 'CSRP3-Related_Disorders', 'Aortic_valve_disorder', 'Levothyroxine_response', 'Cardioacrofacial_dysplasia', 'Familial_hypoalphalipoproteinemia', 'NOS2-related_condition', 'Abdominal_obesity-metabolic_syndrome', '_intestinal_dysrhythmia', 'ADRB2_POLYMORPHISM', 'Atypical_hemolytic-uremic_syndrome_with_thrombomodulin_anomaly', 'MPI-related_condition', 'TREX1-Related_Disorders', 'Neurodevelopmental_disorder_with_craniofacial_dysmorphism_and_skeletal_defects', 'TRIM63-related_condition', 'GNB2-related_condition', 'YARS2-related_condition', 'Hereditary_Sideroblastic_Anemia_with_Myopathy_', 'CRELD1-related_condition', 'Iron_Overload', 'Autosomal_recessive_AGK-related_phenotype', 'APOA1-related_condition', 'NAA15-related_condition', 'Mitochondrial_cytopathy', 'FGF17-related_condition', 'NEBL-related_Cardiomyopathy', 'Sarcoidosis', 'primray_hypomagnesemia_with_secondary_hypocalcemia', 'FIBRINOGEN_KYOTO', 'Chronic_obstructive_pulmonary_disease', 'Disorder_of_cardiovascular_system', 'Intraventricular_hemorrhage', 'Neonatal_cardiomyopathy', 'PRDM16-related_condition', 'SCO2-related_condition', 'warfarin_response_-_Dosage', 'L', 'ITGAM-related_condition', 'Glycogen_storage_disease_IIIa', 'Abnormal_circulating_lipid_concentration', 'IL6ST-related_condition', 'Cardiac', 'Congenital_defect_of_folate_absorption', 'TNNI2-related_condition', 'FHL2-related_condition', 'Hereditary_arterial_', 'RIN2_syndrome', 'LTBP2-Related_Disorders', 'FBLN1-related_condition', 'Coronary_artery_disease', 'Dilated_cardiomyopathy_1U', 'Timothy_syndrome_type', 'Fatal_Infantile_Cardioencephalomyopathy', 'Hereditary_pheochromocytoma-paraganglioma', 'Heart-hand_syndrome', 'Aortic_valve_disease', 'CD55-related_condition', 'Familial_congenital_diaphragmatic_hernia', 'Pseudohypoaldosteronism_type_2B', 'Autosomal_recessive_inherited_pseudoxanthoma_elasticum', 'focal_', 'PI', 'ABCC6-related_disorder', 'EPHX2-related_condition', 'Hypobetalipoproteinemia', 'Hemochromatosis_type_2B', 'NUP155-related_condition', 'CTNNA3-related_condition', 'COL5A2-related_condition', 'TNNT3-related_condition', 'ALDH2-related_condition', 'mitochondrial_hepatopathy', 'PHACTR1-related_condition', 'D', 'Myopia', 'Legius_syndrome', 'Transposition_of_the_great_arteries', 'VCAN-related_condition', 'CAVIN1-related_condition', 'Steinert_myotonic_dystrophy_syndrome', 'Becker_muscular_dystrophy', 'Adams-Oliver_syndrome', 'Bannayan-Riley-Ruvalcaba_syndrome', 'Heart_disease', 'LRP1-related_condition', 'Abnormal_retinal_morphology', 'Progressive_ventriculomegaly', 'Asymmetric_septal_hypertrophy', 'WLS_syndrome', 'APOLIPOPROTEIN_A-IV_RARE_VARIANT', 'PLIN1-related_familial_partial_lipodystrophy', 'X-linked_Opitz_G/BBB_syndrome', 'VANGL2-related_condition', 'CK_syndrome', 'Familial_thoracic_aortic_aneurysm_and_aortic_dissection', 'Motor_', 'LGALS2-related_condition', 'PPARGC1A-related_condition', 'Dystrophin_deficiency', 'ABCC9-related_disorder', 'Ischemic_stroke', 'Obesity', 'ADCY3-related_condition', 'Bardet-Biedl_syndrome_1/7', 'Mandibuloacral_dysplasia', 'SHOX2-related_condition', 'KCND3-related_condition', 'Heterotaxy', 'Conduction_disorder_of_the_heart', 'Familial_cardiomyopathy', 'Marfanoid_habitus_and_intellectual_disability', 'CNBP-related_condition', 'Myotonic_dystrophy_type', 'Cardiac_valvular_defect', 'Dystrophinopathy', 'Exercise_intolerance_', 'PKP1-related_condition', 'APOA5-related_condition', 'Café-au-lait_macules_with_pulmonary_stenosis', 'RAF1-related_condition', 'GDF3-related_condition', 'APOA2-related_condition', 'SCN5A-related_condition', 'King_Denborough_syndrome', 'NOX1-related_condition', 'IL6-related_condition', 'Hypertriglyceridemia', 'Severe_SARS-CoV-2_infection', 'Diabetes_Mellitus', 'KCNE2-Related_Disorders', 'ISL1-related_condition', 'ITGA2-related_condition', 'Cardio-facio-cutaneous_syndrome', 'Intellectual_disability-cardiac_anomalies-short_stature-joint_laxity_syndrome', 'RAP1A-related_condition', 'MYBPC3-related_condition', 'PAPPA2-related_condition', 'DUSP6-related_condition', 'HDAC4-related_condition', 'ABCC6-related_condition', 'KCNH1-related_disorders', 'CDK13-Related_Disorder', 'Mitochondrial_DNA_depletion_syndrome_12B_(cardiomyopathic_type)', 'Arteriosclerosis_disorder', 'Holt-Oram_syndrome', 'EPHB4-associated_vascular_malformation_spectrum', 'Abnormal_cardiovascular_system_morphology', 'ANGPTL3-related_condition', 'NDUFV2-related_condition', 'ABCC9-related_condition', 'PLCZ1-related_condition', 'Protein-losing_enteropathy', 'TANGO2-related_condition', 'HCN4-related_condition', 'X-linked_DMD-related_dystrophinopathy', 'EDN1-related_condition', 'Glycogen_storage_disease_type_IXc', 'Noonan_syndrome-like_disorder_with_juvenile_myelomonocytic_leukemia', '_congenital_anomalies', 'Myocardial_infarction', 'CIB1-related_condition', 'NDUFS7-related_condition', 'CTNNA1-associated_FEVR', 'HANAC-like_syndrome', 'TYPE_2_DIABETES_MELLITUS', 'NPR2-related_condition', 'EEG_abnormality', 'ABCA1_polymorphism', 'Atrial_conduction_disease', 'Williams_syndrome', 'Familial_visceral_amyloidosis', 'MYH7-related_disorder', 'RAB29-related_condition', 'Pulmonary_valve_atresia', 'SMAD2-congenital_heart_disease_and_multiple_congenital_anomaly_disorder', 'Homocystinuria_due_to_methylene_tetrahydrofolate_reductase_deficiency', 'NDUFAF1-related_condition', 'Generalized_arterial_tortuosity', 'FNDC3B-related_condition', 'HMOX1-related_condition', '_heart_glycogen_synthase_deficiency', 'Thrombus', 'Gastrointestinal_defect_', 'Myopathy_due_to_calsequestrin_', 'Atrial_fibrillation', 'Capillary_malformation-arteriovenous_malformation_syndrome', 'Distichiasis-lymphedema_syndrome', 'SCN4B-related_condition', 'Scimitar_anomaly', 'Peripheral_arterial_occlusive_disease', 'Complete_trisomy_21_syndrome', 'Coronary_artery_disease/myocardial_infarction', 'Right_atrial_isomerism', 'CHIME_syndrome', 'Coarctation_of_aorta', 'Abetalipoproteinaemia']
class RealClinVar:
    def __init__(self, num_records=10, all_records=True):
        self.records = load_real_clinvar()
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='CLASS'):
        data = []
        for record in self.records:
            ref,alt = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
            if ref is None:
                continue
            x = [ref, alt, 0]
            if target=='CLASS':
                y = record['class']
            elif target=='PHENOTYPE':
                y = record['phenotype']
                if y == '.':
                    continue
            data.append([x, y])
        return data






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

class MAVEDataWrapper:
    def __init__(self, num_records=2000, all_records=False,
                 # Essential filters for training stability
                 filter_genes=None, experimental_methods=None,
                 region_type='all', seq_length_range=None, max_studies=None,
                 variant_types=None):
        self.num_records = num_records
        self.all_records = all_records
        self.max_studies = max_studies  # Maximum number of studies to process
        # Essential filters
        self.filter_genes = filter_genes  # List of gene names to filter
        self.experimental_methods = experimental_methods  # List of experimental methods
        self.region_type = region_type  # 'coding', 'non-coding', or 'all'
        self.seq_length_range = seq_length_range  # Tuple (min_len, max_len)
        self.variant_types = variant_types  # List of variant types to include (e.g., ['sub', 'del', 'ins'])

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def filter_data(self, data):
        """Apply essential filtering to MAVES data."""
        original_size = len(data)
        filtered_data = []

        for record in data:
            annotation = record[0][2] if isinstance(record[0], list) else record[0].get('annotation', '')
            ref_seq = record[0][0] if isinstance(record[0], list) else record[0].get('ref', '')

            # 1. Gene filter
            if self.filter_genes:
                gene_found = any(gene.upper() in annotation.upper() for gene in self.filter_genes)
                if not gene_found:
                    continue

            # 2. Auto-exclude control samples and processed data
            exclude_patterns = MAVE_METHODS.get("EXCLUDE_FROM_TRAINING", [])
            if any(pattern.lower() in annotation.lower() for pattern in exclude_patterns):
                continue

            # 3. Experimental method filter (supports both individual methods and categories)
            if self.experimental_methods:
                all_methods = expand_method_filters(self.experimental_methods)
                method_found = any(method.lower() in annotation.lower() for method in all_methods)
                if not method_found:
                    continue

            # 4. Region type filter (coding/non-coding)
            if self.region_type != 'all':
                # Extract HGVS prefix to determine coding vs non-coding
                if ', HGVS Prefix: ' in annotation:
                    hgvs_prefix = annotation.split(', HGVS Prefix: ')[1].split(',')[0].strip()
                    is_coding = hgvs_prefix == 'c'

                    if self.region_type == 'coding' and not is_coding:
                        continue
                    elif self.region_type == 'non-coding' and is_coding:
                        continue

            # 5. Sequence length filter
            if self.seq_length_range:
                min_len, max_len = self.seq_length_range
                seq_len = len(ref_seq)
                if not (min_len <= seq_len <= max_len):
                    continue

            # 6. Variant type filter
            if self.variant_types:
                # Extract variant type from annotation string
                if 'Variant Type: ' in annotation:
                    variant_type = annotation.split('Variant Type: ')[1].split(',')[0].strip()
                    if variant_type not in self.variant_types:
                        continue
                else:
                    # Skip if variant type not found and filter is specified
                    continue

            filtered_data.append(record)


        print(f"Filtered data: {original_size:,} -> {len(filtered_data):,} records")
        excluded_count = 0
        if original_size > 0:
            excluded_count = original_size - len(filtered_data)
            exclusion_rate = excluded_count / original_size * 100
            print(f"  Excluded: {excluded_count:,} records ({exclusion_rate:.1f}%)")

        if self.filter_genes:
            print(f"  Gene filter: {self.filter_genes}")
        if self.experimental_methods:
            print(f"  Experimental methods: {self.experimental_methods}")
        if self.region_type != 'all':
            print(f"  Region type: {self.region_type}")
        if self.seq_length_range:
            print(f"  Sequence length range: {self.seq_length_range}")
        if self.variant_types:
            print(f"  Variant types: {self.variant_types}")

        exclude_patterns = MAVE_METHODS.get("EXCLUDE_FROM_TRAINING", [])
        if exclude_patterns:
            print(f"  Auto-excluded patterns: {', '.join(exclude_patterns)}")

        return filtered_data

    def get_data(self, Seq_length=20, target='score', sequence_type='dna', region_type=None):
        if os.path.exists('./root/data/maves.jsonl'):
            data = read_jsonl('./root/data/maves.jsonl')
        else:
            if region_type is not None:
                file_name = f'./root/data/maves_{sequence_type}_{region_type}_{Seq_length}.jsonl'
            else:
                file_name = f'./root/data/maves_{sequence_type}_{Seq_length}.jsonl'
            data = get_maves(seq_length=Seq_length, limit=self.max_studies, target=target, sequence_type=sequence_type, region_type=region_type)
            save_as_jsonl(data, file_name)

        # Apply filters
        data = self.filter_data(data)

        if self.all_records:
            return data

        return data[:self.num_records]

class GWASDataWrapper:
    def __init__(self, num_records=2000, all_records=True):
        self.num_records = num_records
        self.gwas_catalog = download_file(file_path='./root/data/gwas_catalog_v1.0.2-associations_e111_r2024-03-01.tsv', gwas_path='alternative')
        self.trait_mappings = download_file(file_path='./root/data/gwas_catalog_trait-mappings_r2024-03-01.tsv', gwas_path='trait_mappings')
        self.all_records = all_records
        self.genome_extractor = GenomeSequenceExtractor()

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, target='P-Value'):
        # return (x, y) pairs
        if os.path.exists('./root/data/gwas.jsonl'):
            data = read_jsonl('./root/data/gwas.jsonl')
        else:
            data = []
            disease_to_efo = self.trait_mappings.set_index('Disease trait')['EFO term'].to_dict()
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
                            save_as_jsonl(data,'./root/data/gwas.jsonl')
        if self.all_records:
            return data
        return data[:self.num_records]

class ClinVarDataWrapper:
    def __init__(self, num_records=30000, all_records=True, use_default_dir=True):
        if use_default_dir:
            self.clinvar_vcf_path = load_clinvar.download_file()
            self.genome_extractor = GenomeSequenceExtractor()
        else:
            self.clinvar_vcf_path = load_clinvar.download_file(vcf_file_path='./root/data/clinvar_20250409.vcf') # for local use
            self.genome_extractor = GenomeSequenceExtractor('./root/data/hg19.fa') # for amlt
        self.records = load_clinvar.read_vcf(self.clinvar_vcf_path,
                                             num_records=num_records,
                                             all_records=all_records)


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
    def parse_pathg(self,y):
        if y[0] == "Uncertain_significance" or y[0] == "Conflicting_classifications_of_pathogenicity"   or    y[0] == "not_provided":
            return None
        if y[0] == 'Pathogenic/Likely_pathogenic':
            y[0] = 'Likely_pathogenic'
        if y[0] == 'Benign/Likely_benign':
            y[0] = 'Likely_benign'
        # if y[0] not in ['Benign', 'Likely_benign', 'Likely_pathogenic', 'Pathogenic']:
        if y[0] not in ['Benign', 'Pathogenic']:
            return None
        # map 'Benign', 'Likely_benign' to Benign, 'Likely_pathogenic', 'Pathogenic' to Pathogenic
        if y[0] == 'Benign' or y[0] == 'Likely_benign':
            y[0] = 'Benign'
        elif y[0] == 'Likely_pathogenic' or y[0] == 'Pathogenic':
            y[0] = 'Pathogenic'
        return y[0]
    def get_data(self, Seq_length=20, target='CLNSIG', disease_subset=False):
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
            elif target == 'CLNDN' or target=='DISEASE_PATHOGENICITY':
                # if len(record['CLNDN'])<= 1:
                    # print(record['CLNDN'])
                    # data.append([x, self.convert_disease_name(disease)])
                if len(record['CLNDN'])>=1 and record['CLNDN'][0] is not None:
                    y = record['CLNDN'][0].split('|') # disease name
                    # each mutation can be associated with multiple diseases
                    # but most of those diseases are related, so we just take the first one which is not 'not_provided'
                    for disease in y:
                        if disease != 'not_provided':
                            if target=='DISEASE_PATHOGENICITY':
                                data.append([x, (self.convert_disease_name(disease),self.parse_pathg(record['CLNSIG']))])
                            else:
                                data.append([x, self.convert_disease_name(disease)])
                            break
        if disease_subset and target == 'DISEASE_PATHOGENICITY':
            data_subset = []
            for i in range(len(data)):
                if data[i][1][0] in DISEASE_SUBSET and data[i][1][1] is not None:
                    data_subset.append([data[i][0],data[i][1][1]])
            return data_subset
        if disease_subset and target == 'CLNDN':
            data_subset = []
            # get the length of records with the disease subset
            for i in range(len(data)):
                if data[i][1] in DISEASE_SUBSET:
                    data[i][1]  = DISEASE_SUBSET[0]
                    data_subset.append(data[i])
            num_record_with_dis_subset = len(data_subset)
            # get the same number of other records
            for i in range(len(data)):
                if num_record_with_dis_subset == 0:
                    break
                if data[i][1] not in DISEASE_SUBSET:
                    data[i][1] = 'Other_disease'
                    data_subset.append(data[i])
                num_record_with_dis_subset -= 1
            return data_subset
        return data


class ClinVarDataWrapperPrintPercent:
    def __init__(self, num_records=30000, all_records=True, use_default_dir=True):
        if use_default_dir:
            self.clinvar_vcf_path = load_clinvar.download_file()
            self.genome_extractor = GenomeSequenceExtractor()
        else:
            self.clinvar_vcf_path = load_clinvar.download_file(vcf_file_path='./root/data/clinvar_20250409.vcf') # for local use
            self.genome_extractor = GenomeSequenceExtractor('./root/data/hg19.fa') # for amlt
        self.records = load_clinvar.read_vcf(self.clinvar_vcf_path,
                                             num_records=num_records,
                                             all_records=all_records)


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

    def parse_pathg(self,y):
        if y[0] == "Uncertain_significance" or y[0] == "Conflicting_classifications_of_pathogenicity"   or    y[0] == "not_provided":
            return None
        if y[0] == 'Pathogenic/Likely_pathogenic':
            y[0] = 'Likely_pathogenic'
        if y[0] == 'Benign/Likely_benign':
            y[0] = 'Likely_benign'
        # if y[0] not in ['Benign', 'Likely_benign', 'Likely_pathogenic', 'Pathogenic']:
        if y[0] not in ['Benign', 'Pathogenic']:
            return None
        # map 'Benign', 'Likely_benign' to Benign, 'Likely_pathogenic', 'Pathogenic' to Pathogenic
        if y[0] == 'Benign' or y[0] == 'Likely_benign':
            y[0] = 'Benign'
        elif y[0] == 'Likely_pathogenic' or y[0] == 'Pathogenic':
            y[0] = 'Pathogenic'
        return y[0]

    def get_data(self, Seq_length=20, target='CLNSIG', disease_subset=False):
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
            elif target == 'CLNDN' or target=='DISEASE_PATHOGENICITY':
                # if len(record['CLNDN'])<= 1:
                    # print(record['CLNDN'])
                    # data.append([x, self.convert_disease_name(disease)])
                if len(record['CLNDN'])>=1 and record['CLNDN'][0] is not None:
                    y = record['CLNDN'][0].split('|') # disease name
                    # each mutation can be associated with multiple diseases
                    # but most of those diseases are related, so we just take the first one which is not 'not_provided'
                    for disease in y:
                        if disease != 'not_provided':
                            if target=='DISEASE_PATHOGENICITY':
                                data.append([x, (self.convert_disease_name(disease),self.parse_pathg(record['CLNSIG']))])
                            else:
                                data.append([x, self.convert_disease_name(disease)])
                            break

        if disease_subset and target == 'DISEASE_PATHOGENICITY':
            data_subset = []
            benign_count = 0
            total_count = 0

            for i in range(len(data)):
                if data[i][1][0] in DISEASE_SUBSET :
                    data_subset.append([data[i][0],data[i][1][1]])
                    total_count += 1
                    if data[i][1][1] is not None and (data[i][1][1] == 'Benign' or data[i][1][1] == 'Likely_benign'):
                        benign_count += 1

            if total_count > 0:
                benign_percentage = (benign_count / total_count) * 100
                print(f"Percentage of benign samples in disease subset: {benign_percentage:.2f}% ({benign_count}/{total_count})")
            else:
                print("No samples found in disease subset")

            return data_subset,benign_percentage

        if disease_subset and target == 'CLNDN':
            data_subset = []
            # get the length of records with the disease subset
            for i in range(len(data)):
                if data[i][1] in DISEASE_SUBSET:
                    data[i][1]  = DISEASE_SUBSET[0]
                    data_subset.append(data[i])
            num_record_with_dis_subset = len(data_subset)
            # get the same number of other records
            for i in range(len(data)):
                if num_record_with_dis_subset == 0:
                    break
                if data[i][1] not in DISEASE_SUBSET:
                    data[i][1] = 'Other_disease'
                    data_subset.append(data[i])
                num_record_with_dis_subset -= 1
            return data_subset
        return data,benign_percentage


class GeneKoDataWrapper:
    def __init__(self, num_records=1000, all_records=True):
        self.num_records = num_records
        self.fitness_scores = create_fitness_scores_dataframe()
        self.genome_extractor = GenomeSequenceExtractor()
        if all_records:
            self.num_records = len(self.fitness_scores)
        print(f"Number of records: {self.num_records} out of {len(self.fitness_scores)}")

    def __call__(self, *args: Any) -> Any:
        return self.get_data(*args)

    def get_data(self, Seq_length=20, insert_Ns=True):
        # return (x, y) pairs
        data = []
        for i in tqdm(range(self.num_records)):
            gene = self.fitness_scores.iloc[i]
            record = create_variant_sequence_and_reference_sequence_for_gene(gene, insert_Ns=insert_Ns)
            ref, alt = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
            if ref is None:
                continue
            cell_line, cell_line_score = self.flatten(gene)

            x = [ref, alt, cell_line[CELL_LINE]]
            y = cell_line_score[CELL_LINE]
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
            num_pos = 0
            for i in range(self.num_records):
                row = records.iloc[i]
                record = row['record']
                slop = row['slope']
                p_val = row['pval_nominal']
                p_threshold = row['pval_nominal_threshold']
                reference, alternate = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
                if reference is None:
                    continue
                x = [reference, alternate, organism]
                if target == 'slope':
                    if slop < 0:
                        y = "negative"
                    else:
                        y = "positive"
                        num_pos += 1
                elif target == 'p_val':
                    if p_val < p_threshold:
                        y = "significant"
                    else:
                        y = "not_significant"
                data.append([x, y])
            print(f"Number of positive examples: {num_pos} out of {self.num_records} total")
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
            num_pos = 0
            for i in range(self.num_records):
                row = records.iloc[i]
                record = row['record']
                splice_position = row['phenotype_id'].split(':')
                splice_position_distance_change = int(splice_position[2]) - int(splice_position[1])
                slop = row['slope']
                p_val = row['pval_nominal']
                p_threshold = row['pval_nominal_threshold']
                reference, alternate = self.genome_extractor.extract_sequence_from_record(record, sequence_length=Seq_length)
                if reference is None:
                    continue
                x = [reference, alternate, organism]
                if target == 'slope':
                    if slop < 0:
                        y = "negative"
                    else:
                        y = "positive"
                        num_pos += 1
                    # y = slop
                elif target == 'p_val':
                    if p_val < p_threshold:
                        y = "significant"
                    else:
                        y = "not_significant"
                    # y = p_val
                elif target == 'splice_change':
                    if splice_position_distance_change < 0:
                        y = "negative"
                    else:
                        y = "positive"
                    # y = splice_position_distance_change
                data.append([x, y])
            print(f"Number of positive examples: {num_pos} out of {self.num_records} total")
        return data




def create_variant_record(
    chrom: Union[str, int],
    pos: int,
    ref: str,
    alt: str,
    variant_id: Union[str, int] = 0,
) -> dict[str, Any]:
    """
    Build a lightweight record **exactly** matching the keys consumed by
    `GenomeSequenceExtractor.extract_sequence_from_record`.
    Example output
    --------------
    {
        "Chromosome": "chr7",
        "Position": 140_434_574,
        "Reference Base": "C",
        "Alternate Base": ("T",),
        "ID": "42",
    }
    """
    chrom = str(chrom)
    if not chrom.lower().startswith("chr"):
        chrom = f"chr{chrom}"

    alt_first = str(alt).split(",")[0].strip().upper()

    record = {
        "Chromosome": chrom,
        "Position": int(pos),
        "Reference Base": str(ref).upper(),
        "Alternate Base": (alt_first,),           # tuple → safe `[0]`
        "ID": str(variant_id),
    }
    return record


class SmartVariantDataWrapper:
    def __init__(
        self,
        csv_path: str | Path,
        num_records: int = 1_000,
        all_records: bool = False,
        fasta_path: str | Path | None = None,
        gtf_path: str | None = None,
    ):
        self.variants = pd.read_csv(csv_path, low_memory=False)
        if all_records:
            num_records = len(self.variants)
        self.num_records = min(num_records, len(self.variants))
        self.genome_extractor = GenomeSequenceExtractor('./root/data/hg19.fa')
        print(
            f"Number of records: {self.num_records} "
            f"out of {len(self.variants)} in '{csv_path}'."
        )

    # -------------------------------------------------------------- #
    __call__ = lambda self, *a, **kw: self.get_data(*a, **kw)
    # -------------------------------------------------------------- #
    def get_data(
        self,
        Seq_length: int = 20,
        insert_Ns: bool = True,
        progress_bar: bool = True,
        target: str = 'score',
        threshold: float = 50.0,
        min_samples_per_class: int = 2,
    ) -> List[Tuple[Tuple[str, str], Union[float, str]]]:
        data: list[tuple[tuple[str, str], Union[float, str]]] = []

        if target == 'disease':
            # First pass: collect all samples by disease class
            class_samples = {
                'Aortopathy': [],
                'Cardiomyopathy': [],
                'Arrhythmia': [],
                'Structural defect': []
            }

            iterator = range(self.num_records)
            if progress_bar:
                iterator = tqdm(iterator, desc="extracting variant contexts")

            for idx in iterator:
                row = self.variants.iloc[idx]
                disease_class = self._extract_disease_class(row)

                if disease_class is not None:
                    smart_score = float(row["smart_score"])
                    record = create_variant_record(
                        chrom=row["CHROM"],
                        pos=int(row["start"]),
                        ref=row["ref_allele"],
                        alt=row["alt_allele"],
                        variant_id=idx,
                    )
                    ref_seq, alt_seq = self.genome_extractor.extract_sequence_from_record(
                        record, sequence_length=Seq_length
                    )
                    if ref_seq is not None and alt_seq is not None:
                        class_samples[disease_class].append(([ref_seq, alt_seq, None], disease_class, smart_score))

            # Second pass: apply threshold with minimum samples per class
            print(f"\nClass distribution before threshold filtering:")
            for cls, samples in class_samples.items():
                print(f"  {cls}: {len(samples)}")

            # For each class, keep samples above threshold OR top N samples if below min_samples_per_class
            for cls, samples in class_samples.items():
                # Sort by SMART score descending
                samples.sort(key=lambda x: x[2], reverse=True)

                # Filter by threshold
                filtered = [s for s in samples if s[2] >= threshold]

                # If too few samples, take top N regardless of threshold
                if len(filtered) < min_samples_per_class and len(samples) > 0:
                    print(f"Warning: {cls} has only {len(filtered)} samples >= {threshold}. Taking top {min(min_samples_per_class, len(samples))}.")
                    filtered = samples[:min(min_samples_per_class, len(samples))]

                # Add to data (drop the smart_score)
                data.extend([(x[0], x[1]) for x in filtered])

            print(f"\nClass distribution after threshold filtering (threshold={threshold}):")
            from collections import Counter
            class_counts = Counter([x[1] for x in data])
            for cls in ['Aortopathy', 'Cardiomyopathy', 'Arrhythmia', 'Structural defect']:
                print(f"  {cls}: {class_counts.get(cls, 0)}")

        else:  # target == 'score' or pathogenicity
            # Original logic for non-disease targets
            iterator = range(self.num_records)
            if progress_bar:
                iterator = tqdm(iterator, desc="extracting variant contexts")

            for idx in iterator:
                row = self.variants.iloc[idx]
                record = create_variant_record(
                    chrom=row["CHROM"],
                    pos=int(row["start"]),
                    ref=row["ref_allele"],
                    alt=row["alt_allele"],
                    variant_id=idx,
                )
                ref_seq, alt_seq = self.genome_extractor.extract_sequence_from_record(
                    record, sequence_length=Seq_length
                )
                if ref_seq is not None and alt_seq is not None:
                    y = float(row["smart_score"])
                    data.append(([ref_seq, alt_seq, None], y))

        return data
    
    def _extract_disease_class(self, row):
        """
        Extract disease class from patient data.
        Disease classes: Aortopathy, Cardiomyopathy, Arrhythmia, Structural defect
        """
        # Primary method: use disease_class column if it exists
        if 'disease_class' in row and pd.notna(row['disease_class']):
            disease_class = str(row['disease_class']).strip()
            # Return the disease class as-is (already normalized from linking script)
            if disease_class in ['Aortopathy', 'Cardiomyopathy', 'Arrhythmia', 'Structural defect']:
                return disease_class
        
        # No disease class found
        return None
