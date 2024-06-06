# genomic-FM


## Data Set Description
<!-- A table -->
All the tasks are formulated as a classification problem or regression problem. The input is a tuple of `(ref, alt, feature)` where `ref` and `alt` are the reference and alternative alleles, respectively, and `feature` is the additional information about the variant. The output is the label of the variant. The size of the data set is the number of samples in the data set.

| Data Set     | Description                                          | Feature                | Label                             | Size   |
|--------------|------------------------------------------------------|------------------------|-----------------------------------|--------|
| CliVar_CLNSIG       | Genetic Variant Effect Prediction                          | (ref, alt, variant type) | Ourcome Classification: 4 classes   {'Likely_benign': 714866, 'Benign': 195030, 'Pathogenic': 143348, 'Likely_pathogenic': 100859}    | 1,154,103 variants    |
| ClinVar_CLNREVSTAT  | Genetic Variant Disease Prediction                          | (ref, alt, review status) | Disease Type Classification: 13,209 diseases types  | 1,739,691    |
| CellPassport | Cell-type Specific Genetic Variants Prediction    | (ref, alt, celline type) | Classification: 2 classes {'DRV': 1334, 'NPGL': 740404} | 741,738 variants |
| GeneKO       | Gene Knockout Prediction                             | (ref, alt, celline type) | Fitness score, 17,548 mutations in 1107 cellines             | 17548*1107 =  19,425,636 variants   |
| sQTLs        | Splicing Quantitative Trait Loci Prediction          | (ref, alt, organism) | Splicing change/p-val/slope      |  618,932 variants   |
| eQTLs        | Expression Quantitative Trait Loci Prediction        | (ref, alt, organism) | Expression p-val/slope           | 1,207,976 variants    |
| GWAS        | Genome-Wide Association Study Prediction        | (ref, alt, trait) | Expression p-val/beta/odds ratio           | 306,890 SNPs, 53,933 traits/diseases   |
| MAVEs        | Multiplex Assays of Variant Effect Prediction        | (ref, alt, annotation) | Variant Effect Score           | 3,166,541/6,456,426 variants, 1304/1373 studies    |
| Oligogenic Variants       | Oligogenic Variant Effect Prediction        | (ref, alt, disease) | Classification: 2 classes           | 1808 variants combinations, 219 diseases    |


## Draft

1. Variant Prediction
    - Variants Type Prediction with supervised learning
    Task: 9 different tasks

2. Variant Representation with Genomic LLM
    - raw
    - finetuned
    metric: within class distance vs between class distance

3. Variant Indexing
    - Indexing of Variants
    metric: Query time, Accuracy (defined by classes)


* Context Length
* Number of Layers
* Type of heads


## List of Running Experiment
