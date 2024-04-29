# genomic-FM


## Data Set Description
<!-- A table -->
All the tasks are formulated as a classification problem or regression problem. The input is a tuple of `(ref, alt, feature)` where `ref` and `alt` are the reference and alternative alleles, respectively, and `feature` is the additional information about the variant. The output is the label of the variant. The size of the data set is the number of samples in the data set.

| Data Set     | Description                                          | Feature                | Label                             | Size   |
|--------------|------------------------------------------------------|------------------------|-----------------------------------|--------|
| CliVar_CLNSIG       | Genetic Variant Effect Prediction                          | (ref, alt, variant type) | Ourcome Classification: 4 classes   {'Likely_benign': 714866, 'Benign': 195030, 'Pathogenic': 143348, 'Likely_pathogenic': 100859}    | 1,154,103    |
| ClinVar_CLNREVSTAT  | Genetic Variant Disease Prediction                          | (ref, alt, review status) | Disease Type Classification: 13209 diseases types  | 1,739,691    |
| CellPassport | Cell-type Specific Genetic Variants Prediction    | (ref, alt, celline type) | Classification: 2 classes {'DRV': 1334, 'NPGL': 740404} | 741738 |
| GeneKO       | Gene Knockout Prediction                             | (ref, alt, celline type) | Fitness score, 17548 mutations in 1107 cellines             | 17548*1107 =  19,425,636   |
| sQTLs        | Splicing Quantitative Trait Loci Prediction          | (ref, alt, organism) | Splicing change/p-val/slope      |  618,932 mutations    |
| eQTLs        | Expression Quantitative Trait Loci Prediction        | (ref, alt, organism) | Expression p-val/slope           | 1,207,976    |
| GWAS        | Genome-Wide Association Study Prediction        | (ref, alt, trait) | Expression p-val/beta/odds ratio           | 306,890 SNPs    |
| Ensembl Regulatory        | Regulatory Feature Prediction        | (ref, alt, species) | Classification: 2 classes           | 9 species {'Enhancer': 0,'TF_binding_site': 0,'CTCF_binding_site': 0, 'Open_chromatin_region': 0}    |
| MAVEs        | Multiplex Assays of Variant Effect Prediction        | (ref, alt, species) | Variant Effect Score           | 1373 studies    |
| Promoter        | Promoter Prediction        | (ref, alt, organism) | Classification: 2 classes           | 15 species {'Apis mellifera': 6493, 'Homo sapiens': 29598}   |
| Oligogenic Variants       | Oligogenic Variant Effect Prediction        | (ref, alt, disease) | Classification: 2 classes           | 1808 variants combinations, 219 diseases    |
