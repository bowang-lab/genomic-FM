# genomic-FM


## Data Set Description
<!-- A table -->
All the tasks are formulated as a classification problem or regression problem. The input is a tuple of `(ref, alt, feature)` where `ref` and `alt` are the reference and alternative alleles, respectively, and `feature` is the additional information about the variant. The output is the label of the variant. The size of the data set is the number of samples in the data set.

| Data Set     | Description                                          | Feature                | Label                             | Size   |
|--------------|------------------------------------------------------|------------------------|-----------------------------------|--------|
| CliVar_CLNSIG       | Genetic Variant Effect Prediction                          | (ref, alt, variant type) | Ourcome Classification: 4 classes   {'Likely_benign': 714866, 'Benign': 195030, 'Pathogenic': 143348, 'Likely_pathogenic': 100859}    | 1,154,103    |
| ClinVar_CLNREVSTAT  | Genetic Variant Disease Prediction                          | (ref, alt, review status) | Disease Type Classification: 13,209 diseases types  | 1,739,691    |
| CellPassport | Cell-type Specific Genetic Variants Prediction    | (ref, alt, celline type) | Classification: 2 classes {'DRV': 1334, 'NPGL': 740404} | 741,738 |
| GeneKO       | Gene Knockout Prediction                             | (ref, alt, celline type) | Fitness score, 17,548 mutations in 1107 cellines             | 17548*1107 =  19,425,636   |
| sQTLs        | Splicing Quantitative Trait Loci Prediction          | (ref, alt, organism) | Splicing change/p-val/slope      |  618,932 mutations    |
| eQTLs        | Expression Quantitative Trait Loci Prediction        | (ref, alt, organism) | Expression p-val/slope           | 1,207,976    |
| GWAS        | Genome-Wide Association Study Prediction        | (ref, alt, trait) | Expression p-val/beta/odds ratio           | 306,890 SNPs, 53,933 traits/diseases   |
| MAVEs        | Multiplex Assays of Variant Effect Prediction        | (ref, alt, annotation) | Variant Effect Score           | 1,336,464/2,392,753 variants, 500/1373 studies    |
| Oligogenic Variants       | Oligogenic Variant Effect Prediction        | (ref, alt, disease) | Classification: 2 classes           | 1808 variants combinations, 219 diseases    |
| Promoter        | Promoter Prediction        | (species, gene, sequence) | Classification: 2 classes           | 15 species {'Apis mellifera': 6493, 'Arabidopsis thaliana': 22703, 'Caenorhabditis elegans': 7120, 'Canis familiaris': 7545, 'Danio rerio':10728, 'Drosophila melanogaster': 16972, 'Gallus gallus': 6127, 'Homo sapiens': 29598, 'Homo sapiens (non-coding)': 2339, 'Macaca mulatta': 9575 , 'Mus musculus': 25111, 'Mus musculus (non-coding)': 3077,'Plasmodium falciparum': 5597, 'Rattus Norvegicus': 12601, 'Saccharomyces cerevisiae': 5117, 'Schizosaccharomyces pombe': 4802, 'Zea mays': 17081}   |
| Ensembl Regulatory        | Regulatory Feature Prediction        | (species, feature, sequence) | Classification: 2 classes           | 9 species {'Cyprinus carpio carpio': {'Enhancer': 25144, 'Open_chromatin_region': 87976}, 'Dicentrarchus labrax': {'Enhancer': 47062, 'Open_chromatin_region': 67229}, 'Gallus gallus': {'Enhancer': 65118, 'Open_chromatin_region': 171525}, 'Homo sapiens': {'Enhancer': 268483, 'TF_binding_site': 0,'CTCF_binding_site': 0, 'Open_chromatin_region': 0},'Mus musculus': {'Enhancer': 149202, 'TF_binding_site': 17486,'CTCF_binding_site': 50953, 'Open_chromatin_region': 61964}, 'Scophthalmus maximus': {'Enhancer': 24835, 'Open_chromatin_region': 40148}, 'Sus scrofa': {'Enhancer': 134914, 'Open_chromatin_region': 119253}, 'Salmo salar': {'Enhancer': 164105,'Open_chromatin_region': 156157},'Oncorhynchus mykiss': {'Enhancer': 102440, 'Open_chromatin_region': 96880}}    |
