# genomic-FM


## Data Set Description
<!-- A table -->
All the tasks are formulated as a classification problem or regression problem. The input is a tuple of `(ref, alt, feature)` where `ref` and `alt` are the reference and alternative alleles, respectively, and `feature` is the additional information about the variant. The output is the label of the variant. The size of the data set is the number of samples in the data set.

| Data Set     | Description                                          | Feature                | Label                             | Size   |
|--------------|------------------------------------------------------|------------------------|-----------------------------------|--------|
| CliVar_CLNSIG       | Genetic Variants Prediction                          | (ref, alt, variance type) | Ourcome Classification: 4 classes   {'Likely_benign': 714866, 'Benign': 195030, 'Pathogenic': 143348, 'Likely_pathogenic': 100859}    | 1,154,103    |
| ClinVar_CLNREVSTAT  | Genetic Variants Prediction                          | (ref, alt, review status) | Disease Type Classification: 13209 diseases types  | 1,739,691    |
| CellPassport | Experimental verified Genetic Variants Prediction    | (ref, alt, cellline type) | Classification: 2 classes {'DRV': 1334, 'NPGL': 740404} | 741738 |
| GeneKO       | Gene Knockout Prediction                             | (ref, alt, celline type) | Fitness score, 17548 mutations in 1107 cellliens             | 17548*1107 =  19,425,636   |
| sQTLs        | Splicing Quantitative Trait Loci Prediction          | (ref, alt, organism) | Splicing change/p-val/slope      |  618,932 mutations    |
| eQTLs        | Expression Quantitative Trait Loci Prediction        | (ref, alt, organism) | Expression p-val/slope           | 1,207,976    |
| GWAS        | Genome-Wide Association Study Prediction        | (ref, alt, trait) | Expression p-val/beta/odds ratio           | 306,890    |
| Ensembl Regulatory        | Regulatory Feature Prediction        | (ref, alt, organism) | Classification: 2 classes           | X    |
| MAVEs        | Multiplex Assays of Variant Effect Prediction        | (ref, alt, organism) | Score           | 84/1373    |
| Promoter        | Promoter Prediction        | (ref, alt, organism) | Classification: 2 classes           | X    |
