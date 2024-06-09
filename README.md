# GV-Rep

This repository contains the GV-Rep dataset and integration with Genomic Foundation Models for model finetuning. Designed for academic research, it features a comprehensive collection of 7 million GV records with detailed annotations and one clinician verified dataset. The dataset supports deep learning models in learning GV representations across various traits and contexts.

### Setup

#### Hardware Requirements

- for dataset download: 30GB of disk space
- for model finetuning: we recommend using a GPU V100 or better

#### Dependencies
First clone the repository
```bash
git clone https://github.com/bowang-lab/genomic-FM.git
```
Next, move to the GV-Rep directory
```bash
cd genomic-FM
```
Then, install the dependencies


```bash
pip install torch==2.2.0 torchvision
pip install -r requirements.txt
```

Finall add  the submodule if you want to use the Genomic Foundation Models
```bash
git submodule update --init --recursive
```
For the use of Indexing functionality, `Faiss` is needed. Please install it using the following command:
```bash
# CPU-only version
$ conda install -c pytorch faiss-cpu=1.8.0

# GPU(+CPU) version
$ conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# GPU(+CPU) version with NVIDIA RAFT
$ conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0
```
### Usage

#### Dataset Download
While you can programmably access all the data, we strongly recommend running the following script to download the
cached raw data files from zenodo. This will save you time and bandwidth.
```bash
python download_data.py
```
Instead you can manually download the data from the Zenodo repository [here](https://zenodo.org/records/11502840). And save it locally in the `genomic-FM/root/data` directory.

#### Accessing the Data

Use the following code snippets to load various datasets. Adjust NUM_RECORDS, ALL_RECORDS, and SEQ_LEN as needed.
```python
from src.dataloader.data_wrapper import (
    RealClinVar, OligogenicDataWrapper, MAVEDataWrapper,
    GWASDataWrapper, ClinVarDataWrapper, GeneKoDataWrapper,
    CellPassportDataWrapper, eQTLDataWrapper, sQTLDataWrapper
)

NUM_RECORDS = 1000
ALL_RECORDS = False
SEQ_LEN = 20

# Load RealClinVar data
data_loader = RealClinVar(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# Load Oligogenic data
data_loader = OligogenicDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# Load ClinVar data
data_loader = ClinVarDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# Load GeneKo data
data_loader = GeneKoDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# Load CellPassport data
data_loader = CellPassportDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# Load eQTL data
data_loader = eQTLDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# Load sQTL data
data_loader = sQTLDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# Load MAVE data
data_loader = MAVEDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)

# Load GWAS data
data_loader = GWASDataWrapper(num_records=NUM_RECORDS, all_records=ALL_RECORDS)
data = data_loader.get_data(Seq_length=SEQ_LEN)
print(data)
```
#### Model Finetuning
Ensure you have 1 GPU available for model finetuning. Note that you should define the config file in yaml format before running the script. And example of the config file ```finetune_dnabert2.yaml``` is shown below:

```yaml
clinvar_CLNSIG_dnabert2:
  class: ClinVarDataWrapper
  task: classification
  target: CLNSIG
  Seq_length: 1024
  pca_components: 16
  model_initiator_name: dnabert2
  output_size: 4

sqtl_pval_dnabert2:
  class: sQTLDataWrapper
  task: classification
  Seq_length: 1024
  pca_components: 16
  target: p_val
  model_initiator_name: dnabert2
  output_size: 2
```


Run the following script to finetune the model. Note that ```--project``` is needed for specifying the project name in wandb. The project name should be unique to avoid conflicts with other users.
```bash
wandb offline # if GPU compute cannot access the internet
python finetune.py --dataset='sqtl_pval_dnabert2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_dnabert2.yaml --seed=0 --project='GV-Rep'
python finetune.py --dataset='clinvar_CLNSIG_dnabert2' --epochs=100 --gpus=8 --num_workers=8 --config=configs/finetune_dnabert2.yaml --seed=0 --project='GV-Rep'
```

#### Genetic Variants Indexing
Similar to fine-tuning, you should define the config file in yaml format before running the script. An example of the config file ```indexing.yaml``` is shown below. ```final_pca_components``` is the number of PCA components used for indexing.

```yaml
clinvar_CLNSIG_hyena-tiny:
  class: ClinVarDataWrapper
  task: classification
  target: CLNSIG
  Seq_length: 1024
  pca_components: 16
  final_pca_components: 8
  model_initiator_name: hyenadna-tiny-1k
  num_records: 40000
  all_records: False
  output_size: 4
```
To index the genetic variants, use the following code snippet. The checkpoint path is the path to the model checkpoint file. The dataset is the dataset name defined in the config file.

```python
from src.variants_vector_index.vector_loader import VectorLoader
import numpy as np
import time

vec_loader = VectorLoader(dataset='clinvar_CLNSIG_hyena-tiny',checkpoint='Run-GFM/luxnk59q/checkpoints/epoch=99-step=431100.ckpt')

query_vector = vec_loader.vectors[1]
query_vector_label = vec_loader.labels[1]
start_time = time.time()
distances, result_labels, indices = vec_loader.query_vectors(query_vector, k=20)
end_time = time.time()
query_time = end_time - start_time

print(f"Query vector label: {query_vector_label}")
print(f"Distances: {distances}")
print(f"Result labels: {result_labels}")
print("========================")
print(f"Query vector size: {query_vector.shape}")
print(f"Query time: {query_time} seconds")
```

### License

GV-Rep is distributed under the CC BY-NC-SA license. Users must follow the original licenses of sub-datasets, detailed below. Most sub-datasets are under CC or CC0 licenses, while Cancer Dependency Map data is for educational use only per its original policy[^1].

The Clinician verified GV set is under the CC BY-SA 4.0 license, and the code is under the MIT license[^2].

- ClinVar: CC0 1.0 license
- GTEx: Creative Commons licenses
- MAVEDB: CC BY-NC-SA 4.0
- GWAS: CC0 1.0 license
- OLIDA: CC BY-NC-SA 4.0

[^1]: [https://depmap.sanger.ac.uk/documentation/data-usage-policy/](https://depmap.sanger.ac.uk/documentation/data-usage-policy/)
[^2]: [https://opensource.org/license/mit/](https://opensource.org/license/mit/)
