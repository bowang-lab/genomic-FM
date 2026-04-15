import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from transformers import PreTrainedTokenizer
import datasets as hf_datasets
import logging
import typing
from typing import Dict, Sequence, List, Tuple, Optional
from ..dataloader.data_wrapper import ClinVarDataWrapper,SmartVariantDataWrapper,MAVEDataWrapper, set_disease_subset_from_file
import random
import numpy as np


# =============================================================================
# Multi-Label Dataset Classes (AlphaGenome-style: all heads evaluated per sample)
# =============================================================================

class MultiLabelDataset(Dataset):
    """
    Dataset where each sample has multiple task labels (AlphaGenome-style).

    For SMART data: Each variant has both CLNDN (disease) and CLNSIG (pathogenicity).
    All heads are evaluated for every sample during training.
    """

    IGNORE_INDEX = -100  # Standard PyTorch ignore index for loss masking

    def __init__(
        self,
        data: List[Dict],
        tokenizer: PreTrainedTokenizer,
        task_configs: Dict[str, Dict],
        seq_length: int = 1024
    ):
        """
        Args:
            data: List of dicts with keys 'ref', 'alt', 'CLNDN', 'CLNSIG'
                  e.g., [{'ref': 'ACGT...', 'alt': 'ACGT...', 'CLNDN': 2, 'CLNSIG': 1}, ...]
            tokenizer: Tokenizer for DNA sequences
            task_configs: Dict mapping task names to config
                  e.g., {'CLNDN': {'num_classes': 4}, 'CLNSIG': {'num_classes': 2}}
            seq_length: Maximum sequence length
        """
        super().__init__()
        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())

        ref_sequences = []
        alt_sequences = []
        self.labels = {task: [] for task in self.task_names}

        for item in data:
            ref_sequences.append(item['ref'])
            alt_sequences.append(item['alt'])

            for task in self.task_names:
                label = item.get(task)
                if label is None:
                    self.labels[task].append(self.IGNORE_INDEX)
                else:
                    self.labels[task].append(label)

        # Tokenize sequences
        ref_output = tokenizer(
            ref_sequences,
            return_tensors="pt",
            padding="longest",
            max_length=seq_length,
            truncation=True,
        )
        alt_output = tokenizer(
            alt_sequences,
            return_tensors="pt",
            padding="longest",
            max_length=seq_length,
            truncation=True,
        )

        self.ref_input_ids = ref_output["input_ids"]
        self.ref_attention_mask = ref_output.get("attention_mask")
        self.alt_input_ids = alt_output["input_ids"]
        self.alt_attention_mask = alt_output.get("attention_mask")

        # Log label statistics
        for task in self.task_names:
            valid_count = sum(1 for l in self.labels[task] if l != self.IGNORE_INDEX)
            logging.info(f"MultiLabelDataset: {task} has {valid_count}/{len(self.labels[task])} valid labels")

    def __len__(self):
        return len(self.ref_input_ids)

    def __getitem__(self, i):
        item = {
            "ref_input_ids": self.ref_input_ids[i],
            "alt_input_ids": self.alt_input_ids[i],
        }

        # Add all task labels
        for task in self.task_names:
            item[f"label_{task}"] = self.labels[task][i]

        if self.ref_attention_mask is not None:
            item["ref_attention_mask"] = self.ref_attention_mask[i]
        if self.alt_attention_mask is not None:
            item["alt_attention_mask"] = self.alt_attention_mask[i]

        return item


class MultiLabelDataCollator:
    """Collator for MultiLabelDataset - handles multiple labels per sample."""

    def __init__(self, tokenizer: PreTrainedTokenizer, task_names: List[str]):
        self.tokenizer = tokenizer
        self.task_names = task_names

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ref_input_ids = []
        alt_input_ids = []
        ref_attention_masks = []
        alt_attention_masks = []

        # Collect labels for each task
        task_labels = {task: [] for task in self.task_names}

        for instance in instances:
            ref_input_ids.append(instance["ref_input_ids"])
            alt_input_ids.append(instance["alt_input_ids"])

            if "ref_attention_mask" in instance:
                ref_attention_masks.append(instance["ref_attention_mask"])
            if "alt_attention_mask" in instance:
                alt_attention_masks.append(instance["alt_attention_mask"])

            for task in self.task_names:
                task_labels[task].append(instance.get(f"label_{task}", -100))

        # Pad sequences
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        alt_input_ids = torch.nn.utils.rnn.pad_sequence(
            alt_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        batch = {
            "ref_input_ids": ref_input_ids,
            "alt_input_ids": alt_input_ids,
        }

        # Convert labels to tensors (classification = long)
        for task in self.task_names:
            batch[f"label_{task}"] = torch.tensor(task_labels[task], dtype=torch.long)

        # Handle attention masks
        if ref_attention_masks:
            batch["ref_attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                ref_attention_masks, batch_first=True, padding_value=0
            )
        else:
            batch["ref_attention_mask"] = ref_input_ids.ne(self.tokenizer.pad_token_id)

        if alt_attention_masks:
            batch["alt_attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                alt_attention_masks, batch_first=True, padding_value=0
            )
        else:
            batch["alt_attention_mask"] = alt_input_ids.ne(self.tokenizer.pad_token_id)

        return batch


def return_multilabel_dataset(
    tokenizer: PreTrainedTokenizer,
    csv_path: str = 'root/data/unfiltered_variants.csv',
    threshold: float = 65.0,
    seq_length: int = 1024,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    all_records: bool = True,
    num_records: Optional[int] = None,
):
    """
    Load SMART dataset with paired CLNDN + CLNSIG labels (AlphaGenome-style).

    Each variant gets BOTH labels:
    - CLNDN: Disease classification (4 classes)
    - CLNSIG: Pathogenicity binary (score >= threshold)

    Returns:
        tuple: (datasets_dict, task_num_classes, task_configs, seq_length)
    """
    from sklearn.model_selection import train_test_split

    tokenizer.model_max_length = seq_length

    # Fixed disease labels for SMART data
    disease_labels = sorted(['Aortopathy', 'Arrhythmia', 'Cardiomyopathy', 'Structural defect'])
    disease_to_id = {label: idx for idx, label in enumerate(disease_labels)}

    # Task configurations
    task_configs = {
        'CLNDN': {'num_classes': len(disease_labels)},
        'CLNSIG': {'num_classes': 2},
    }

    # Load SMART data with both disease and score
    wrapper = SmartVariantDataWrapper(
        csv_path=csv_path,
        num_records=num_records or 0,
        all_records=all_records
    )
    raw_data = wrapper.get_multitask_data(Seq_length=seq_length, threshold=threshold)

    # Convert to multi-label format
    multilabel_data = []
    skipped_no_disease = 0

    for ref, alt, disease, score in raw_data:
        sample = {'ref': ref, 'alt': alt}

        # CLNDN: Disease classification
        if disease in disease_to_id:
            sample['CLNDN'] = disease_to_id[disease]
        else:
            sample['CLNDN'] = None
            skipped_no_disease += 1
            continue  # Skip samples without valid disease label

        # CLNSIG: Pathogenicity (always present for SMART data)
        if score is not None:
            sample['CLNSIG'] = 1 if score >= threshold else 0
        else:
            sample['CLNSIG'] = None

        multilabel_data.append(sample)

    print(f"Loaded {len(multilabel_data)} SMART variants with paired labels")
    print(f"  Skipped {skipped_no_disease} variants without valid disease label")

    # Count label availability
    for task in task_configs:
        valid_count = sum(1 for s in multilabel_data if s.get(task) is not None)
        print(f"  {task}: {valid_count} samples ({100*valid_count/len(multilabel_data):.1f}%)")

    # Stratified split based on CLNDN (disease class)
    stratify_labels = [s['CLNDN'] for s in multilabel_data]

    train_data, temp_data = train_test_split(
        multilabel_data,
        test_size=test_split + val_split,
        stratify=stratify_labels,
        random_state=seed
    )

    temp_stratify = [s['CLNDN'] for s in temp_data]
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_split / (val_split + test_split),
        stratify=temp_stratify,
        random_state=seed
    )

    print(f"Splits → Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Print class distribution
    from collections import Counter
    for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        clndn_counts = Counter(s['CLNDN'] for s in split_data)
        clnsig_counts = Counter(s['CLNSIG'] for s in split_data)
        print(f"  {split_name} - CLNDN: {dict(clndn_counts)}, CLNSIG: {dict(clnsig_counts)}")

    # Create datasets
    datasets = {
        'train': MultiLabelDataset(train_data, tokenizer, task_configs, seq_length),
        'val': MultiLabelDataset(val_data, tokenizer, task_configs, seq_length),
        'test': MultiLabelDataset(test_data, tokenizer, task_configs, seq_length),
    }

    # Task num_classes for trainer
    task_num_classes = {task: cfg['num_classes'] for task, cfg in task_configs.items()}

    return datasets, task_num_classes, task_configs, seq_length


# =============================================================================
# Original single-label dataset classes below
# =============================================================================
def split_train_val(dataset_train, val_split=0.1, seed=42):
    """
    Randomly split dataset into train and validation sets.
    """
    train_len = int(len(dataset_train) * (1.0 - val_split))
    generator = torch.Generator().manual_seed(seed)

    train_indices, val_indices = torch.utils.data.random_split(
        range(len(dataset_train)),
        (train_len, len(dataset_train) - train_len),
        generator=generator
    )

    dataset_train_subset = Subset(dataset_train, train_indices)
    dataset_val_subset = Subset(dataset_train, val_indices)

    return dataset_train_subset, dataset_val_subset

def return_clinvar_multitask_dataset(tokenizer: PreTrainedTokenizer, target='CLNDN', disease_subset_file=None,
                                    seq_length=1024, val_split=0.1, test_split=0.1, seed=42):
    """
    3377987, Train: 6381, Validation: 708, Test: 787
    Load ClinVar datasets for multi-task learning.

    Args:
        tokenizer: PreTrainedTokenizer for tokenizing sequences
        target: Target feature to predict ('CLNDN' for disease names by default, or 'CLNSIG' for clinical significance)
        disease_subset_file: Path to text file containing disease names for filtering (one per line)
        seq_length: Sequence length for extraction
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility

    Returns:
        tuple: (multitask_datasets, task_num_classes, max_seq_len)
    """
    tokenizer.model_max_length = seq_length

    # Load disease subset from file if provided
    if disease_subset_file:
        set_disease_subset_from_file(disease_subset_file)

    # Get ClinVar data
    # clinvar_wrapper = ClinVarDataWrapper(all_records=True) # full run
    clinvar_wrapper = ClinVarDataWrapper(all_records=False, use_default_dir=False, num_records=1000000) # for testing
    data = clinvar_wrapper.get_data(Seq_length=seq_length, target=target, disease_subset=bool(disease_subset_file))
    # Dictionary to store datasets
    multitask_datasets = {}

    # Shuffle data
    random.seed(seed)
    random.shuffle(data)
    # Split data into train, validation, and test sets
    total_size = len(data)
    test_size = int(total_size * test_split)
    val_size = int((total_size - test_size) * val_split)
    train_size = total_size - test_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    print(f"Data splits - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    # Get unique labels and create label mapping
    if target == 'CLNDN':
        task_name = 'CLNDN'
        all_labels = sorted(set(item[1] for item in data))
        label_to_id = {label: idx for idx, label in enumerate(all_labels)}
        num_labels = len(all_labels)
        print(f"Found {num_labels} unique disease labels")
    elif target in ['CLNSIG', 'DISEASE_PATHOGENICITY']:
        task_name = target
        # all_labels = ['Benign', 'Likely_benign', 'Likely_pathogenic', 'Pathogenic']
        # mapping Likely_benign and  Benign to Benign; mapping Likely_pathogenic and Pathogenic to Pathogenic
        all_labels = ['Benign', 'Pathogenic']
        label_to_id = {label: idx for idx, label in enumerate(all_labels)}
        print(label_to_id)
        label_to_id = {'Benign': 0, 'Pathogenic': 1, 'Likely_benign': 0, 'Likely_pathogenic': 1 }
        # label_to_id = {'Benign': 0, 'Likely_benign': 0, 'Likely
        num_labels = len(set(label_to_id.values()))
        print(f"Using {num_labels} pathogenicity classes")
    else:
        raise ValueError(f"Unsupported target: {target}")
    # Create datasets (only create val/test if they have data)
    train_dataset = ClinVarDataset(train_data, tokenizer, task_name, label_to_id)
    multitask_datasets['train'] = train_dataset

    if val_data:
        val_dataset = ClinVarDataset(val_data, tokenizer, task_name, label_to_id)
        multitask_datasets[f"{task_name}_val"] = val_dataset

    if test_data:
        test_dataset = ClinVarDataset(test_data, tokenizer, task_name, label_to_id)
        multitask_datasets[f"{task_name}_test"] = test_dataset

    # Track task info
    task_num_classes = {task_name: num_labels}
    max_seq_len = seq_length
    print(f"Max sequence length: {max_seq_len}")

    return multitask_datasets, task_num_classes, max_seq_len

def return_smart_dataset(
    tokenizer: PreTrainedTokenizer,
    csv_path: str,
    target: str = 'score',  # 'disease' for disease classification, 'score' for pathogenicity
    task_name: str = 'CLNDN',  # For dataset naming only
    threshold: float = 65.0,  # For pathogenicity: binarization cutoff; For disease: minimum pathogenicity score
    seq_length: int = 1024,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    all_records: bool = True,
    num_records: int | None = None,
    min_samples_per_class: int = 2,
):
    # ----------------------- load & prepare raw examples -------------------
    tokenizer.model_max_length = seq_length

    wrapper = SmartVariantDataWrapper(
        csv_path=csv_path,
        num_records=num_records or 0,
        all_records=all_records,
    )

    # Get data based on target type with min_samples_per_class parameter
    raw = wrapper.get_data(Seq_length=seq_length, target=target, threshold=threshold, min_samples_per_class=min_samples_per_class)

    if target == 'disease':
        # Disease classification mode
        all_disease_labels = sorted(set(item[1] for item in raw if item[1] is not None))
        label_to_id = {label: idx for idx, label in enumerate(all_disease_labels)}
        num_labels = len(all_disease_labels)

        # Filter out samples without disease labels
        labeled_data = [[seq_pair, label] for seq_pair, label in raw if label is not None and label in label_to_id]

        print(f"Disease classification: Found {num_labels} disease classes: {all_disease_labels}")
        print(f"Total labeled samples: {len(labeled_data)}")

    else:  # target == 'score' (pathogenicity classification)
        # Binarise the target
        labeled_data: List[Tuple[Tuple[str, str], int]] = [
            [seq_pair, 1 if score >= threshold else 0] for seq_pair, score in raw
        ]

        all_disease_labels = [0, 1]  # Binary: benign (0), pathogenic (1)
        label_to_id = {label: idx for idx, label in enumerate(all_disease_labels)}
        num_labels = len(all_disease_labels)

        print(f"Pathogenicity classification: Binary classification with threshold {threshold}")
        print(f"Total samples: {len(labeled_data)}")

    multitask_datasets = {}

    # ------------------------------ stratified splitting ------------------------------
    if target == 'disease':
        # Use stratified split for disease classification to preserve class proportions
        from sklearn.model_selection import train_test_split

        # Extract labels for stratification
        labels = [item[1] for item in labeled_data]

        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            labeled_data,
            test_size=test_split + val_split,
            stratify=labels,
            random_state=seed
        )

        # Second split: val vs test
        temp_labels = [item[1] for item in temp_data]
        val_ratio = val_split / (val_split + test_split)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_ratio),
            stratify=temp_labels,
            random_state=seed
        )

        print(f"Stratified splits → Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        # Print class distribution per split
        from collections import Counter
        for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            split_labels = [item[1] for item in split_data]
            counts = Counter(split_labels)
            print(f"  {split_name} distribution: {dict(counts)}")
    else:
        # For pathogenicity (binary classification), use regular random split
        random.seed(seed)
        random.shuffle(labeled_data)

        total = len(labeled_data)
        test_sz = int(total * test_split)
        val_sz = int((total - test_sz) * val_split)
        train_sz = total - test_sz - val_sz

        train_data = labeled_data[:train_sz]
        val_data = labeled_data[train_sz : train_sz + val_sz]
        test_data = labeled_data[train_sz + val_sz :]

        print(
            f"Smart Variant data → Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )

    # ------------------------------ datasets --------------------------------
    train_ds = ClinVarDataset(train_data, tokenizer,task_name,label_to_id)
    val_ds = ClinVarDataset(val_data, tokenizer,task_name,label_to_id)
    test_ds = ClinVarDataset(test_data, tokenizer,task_name,label_to_id)
    # Store datasets
    multitask_datasets['train'] = train_ds
    multitask_datasets[f"{task_name}_val"] = val_ds
    multitask_datasets[f"{task_name}_test"] = test_ds

    # Track task info
    task_num_classes = {task_name: num_labels}
    max_seq_len = seq_length
    print(f"Max sequence length: {max_seq_len}")

    return multitask_datasets, task_num_classes, max_seq_len


def return_multitask_dataset(
    tokenizer: PreTrainedTokenizer,
    data_source: str = 'smart',  # 'smart' or 'clinvar'
    csv_path: str = 'root/data/unfiltered_variants.csv',
    threshold: float = 65.0,
    seq_length: int = 1024,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
    all_records: bool = True,
    num_records: int | None = None,
    include_clndn: bool = True,
    include_clnsig: bool = True,
    include_maves: bool = False,
    maves_max_samples: int | None = None,
):
    """
    Load dataset for multi-task learning with any combination of CLNDN, CLNSIG, and MAVES.

    Args:
        data_source: 'smart' or 'clinvar' for classification tasks
        include_clndn: Include disease classification task
        include_clnsig: Include pathogenicity classification task
        include_maves: Include MAVES regression task (DMS data)

    Returns:
        tuple: (datasets_dict, task_num_classes, max_seq_len)
    """
    from sklearn.model_selection import train_test_split

    tokenizer.model_max_length = seq_length
    train_datasets = []
    datasets = {}
    task_num_classes = {}

    pathogenicity_to_id = {0: 0, 1: 1}

    # Load classification data from chosen source
    if include_clndn or include_clnsig:
        if data_source == 'clinvar':
            # Use ClinVar data
            clinvar_ds, clinvar_info, _ = return_clinvar_multitask_dataset(
                tokenizer, target='CLNDN' if include_clndn else 'CLNSIG',
                seq_length=seq_length, val_split=val_split, test_split=test_split, seed=seed
            )
            if include_clndn and 'train' in clinvar_ds:
                train_datasets.append(clinvar_ds['train'])
                datasets['CLNDN_val'] = clinvar_ds.get('CLNDN_val')
                datasets['CLNDN_test'] = clinvar_ds.get('CLNDN_test')
                task_num_classes['CLNDN'] = clinvar_info.get('CLNDN', 2)
            if include_clnsig:
                clnsig_ds, clnsig_info, _ = return_clinvar_multitask_dataset(
                    tokenizer, target='CLNSIG', seq_length=seq_length, seed=seed
                )
                if 'train' in clnsig_ds:
                    train_datasets.append(clnsig_ds['train'])
                datasets['CLNSIG_val'] = clnsig_ds.get('CLNSIG_val')
                datasets['CLNSIG_test'] = clnsig_ds.get('CLNSIG_test')
                task_num_classes['CLNSIG'] = 2
        else:
            # Use SMART data
            disease_labels = sorted(['Aortopathy', 'Arrhythmia', 'Cardiomyopathy', 'Structural defect'])
            disease_to_id = {label: idx for idx, label in enumerate(disease_labels)}

            wrapper = SmartVariantDataWrapper(csv_path=csv_path, num_records=num_records or 0, all_records=all_records)
            raw_data = wrapper.get_multitask_data(Seq_length=seq_length, threshold=threshold)

            if include_clndn:
                # Format: ([ref, alt, None], label) - item[0] must be a list with 3 elements
                clndn_data = [([r, a, None], d) for r, a, d, s in raw_data if d in disease_to_id]
                labels = [x[1] for x in clndn_data]
                train, temp = train_test_split(clndn_data, test_size=test_split + val_split, stratify=labels, random_state=seed)
                val, test = train_test_split(temp, test_size=test_split/(val_split+test_split), stratify=[x[1] for x in temp], random_state=seed)
                train_datasets.append(ClinVarDataset(train, tokenizer, 'CLNDN', disease_to_id))
                datasets['CLNDN_val'] = ClinVarDataset(val, tokenizer, 'CLNDN', disease_to_id)
                datasets['CLNDN_test'] = ClinVarDataset(test, tokenizer, 'CLNDN', disease_to_id)
                task_num_classes['CLNDN'] = len(disease_labels)
                print(f"CLNDN → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

            if include_clnsig:
                # Format: ([ref, alt, None], label) - item[0] must be a list with 3 elements
                clnsig_data = [([r, a, None], 1 if s >= threshold else 0) for r, a, d, s in raw_data]
                labels = [x[1] for x in clnsig_data]
                train, temp = train_test_split(clnsig_data, test_size=test_split + val_split, stratify=labels, random_state=seed)
                val, test = train_test_split(temp, test_size=test_split/(val_split+test_split), stratify=[x[1] for x in temp], random_state=seed)
                train_datasets.append(ClinVarDataset(train, tokenizer, 'CLNSIG', pathogenicity_to_id))
                datasets['CLNSIG_val'] = ClinVarDataset(val, tokenizer, 'CLNSIG', pathogenicity_to_id)
                datasets['CLNSIG_test'] = ClinVarDataset(test, tokenizer, 'CLNSIG', pathogenicity_to_id)
                task_num_classes['CLNSIG'] = 2
                print(f"CLNSIG → Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Load MAVES if enabled
    if include_maves:
        maves_ds, maves_info, _ = return_maves_dataset(
            tokenizer, target="score", seq_length=seq_length, seed=seed,
            max_samples_per_experiment=maves_max_samples,
        )
        if 'train' in maves_ds:
            train_datasets.append(maves_ds['train'])
        for k, v in maves_ds.items():
            if k != 'train':
                datasets[k] = v
        task_num_classes.update(maves_info)
        print(f"MAVES added: {maves_info}")

    if not train_datasets:
        raise ValueError("At least one task must be enabled")

    datasets['train'] = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    return datasets, task_num_classes, seq_length


class ClinVarDataset(Dataset):
    """Dataset for ClinVar variant data with separate tokenization for reference and alternative sequences."""

    def __init__(self, data, tokenizer, task_name, label_to_id):
        super(ClinVarDataset, self).__init__()
        self.task_name = task_name
        self.num_labels = len(set(label_to_id.values()))

        # Process data
        ref_sequences = []
        alt_sequences = []
        labels = []
        for item in data:
            # Extract reference and alternative sequences (drop variant_type)
            ref, alt, _ = item[0]
            ref_sequences.append(ref)
            alt_sequences.append(alt)

            # Convert label to ID
            label = label_to_id[item[1]]
            labels.append(label)

        # Tokenize reference sequences
        ref_output = tokenizer(
            ref_sequences,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        # Tokenize alternative sequences
        alt_output = tokenizer(
            alt_sequences,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        # Store tokenized sequences and masks separately
        self.ref_input_ids = ref_output["input_ids"]
        self.ref_attention_mask = ref_output.get("attention_mask")

        self.alt_input_ids = alt_output["input_ids"]
        self.alt_attention_mask = alt_output.get("attention_mask")

        self.labels = labels

    def __len__(self):
        return len(self.ref_input_ids)

    def __getitem__(self, i):
        item = {
            "ref_input_ids": self.ref_input_ids[i],
            "alt_input_ids": self.alt_input_ids[i],
            "labels": self.labels[i],
            "task_name": self.task_name
        }

        if self.ref_attention_mask is not None:
            item["ref_attention_mask"] = self.ref_attention_mask[i]

        if self.alt_attention_mask is not None:
            item["alt_attention_mask"] = self.alt_attention_mask[i]

        return item

class MultiTaskDataCollator:
    """Collate examples for multi-task supervised fine-tuning with separate reference and alternative sequences."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Lists to store separate tensors for reference and alternative sequences
        ref_input_ids, alt_input_ids = [], []
        ref_attention_masks, alt_attention_masks = [], []
        labels = []
        task_names = []

        for instance in instances:
            # Handle reference sequences
            ref_input_ids.append(instance["ref_input_ids"])
            if "ref_attention_mask" in instance:
                ref_attention_masks.append(instance["ref_attention_mask"])

            # Handle alternative sequences
            alt_input_ids.append(instance["alt_input_ids"])
            if "alt_attention_mask" in instance:
                alt_attention_masks.append(instance["alt_attention_mask"])

            # Handle labels and task names
            labels.append(instance["labels"])
            task_names.append(instance.get("task_name", "unknown_task"))

        # Pad reference input_ids
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # Pad alternative input_ids
        alt_input_ids = torch.nn.utils.rnn.pad_sequence(
            alt_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # Convert labels to tensor
        # Use float always to support mixed batches (regression + classification).
        # Classification loss functions must cast to .long() before use.
        is_regression = any("MAVES" in task_name for task_name in task_names)
        if is_regression:
            labels = torch.tensor(labels, dtype=torch.float)
        else:
            labels = torch.tensor(labels, dtype=torch.long)

        # Create output dictionary
        batch = {
            "ref_input_ids": ref_input_ids,
            "alt_input_ids": alt_input_ids,
            "labels": labels,
            "task_names": task_names
        }

        # Handle reference attention masks
        if ref_attention_masks:
            batch["ref_attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                ref_attention_masks, batch_first=True, padding_value=0
            )
        else:
            batch["ref_attention_mask"] = ref_input_ids.ne(self.tokenizer.pad_token_id)

        # Handle alternative attention masks
        if alt_attention_masks:
            batch["alt_attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                alt_attention_masks, batch_first=True, padding_value=0
            )
        else:
            batch["alt_attention_mask"] = alt_input_ids.ne(self.tokenizer.pad_token_id)

        return batch




def return_eqtl_dataset(tokenizer: PreTrainedTokenizer, target='Adipose_Subcutaneous', disease_subset=True,
                                    seq_length=10240, val_split=0.1, test_split=0.1, seed=42):
    # Dictionary to store datasets
    multitask_datasets = {}

    # Load raw datasets
    val_data = hf_datasets.load_dataset(
        "json",
        data_files=f"./root/data/DNALongBench/dnalongbench_hf_{target}/validation.jsonl",
        split="train",
    )
    train_data = hf_datasets.load_dataset(
        "json",
        data_files=f"./root/data/DNALongBench/dnalongbench_hf_{target}/train.jsonl",
        split="train",
    )
    test_data = hf_datasets.load_dataset(
        "json",
        data_files=f"./root/data/DNALongBench/dnalongbench_hf_{target}/test.jsonl",
        split="train",
    )
    # Create datasets
    train_dataset = EqtlDataset(train_data, tokenizer, target)
    val_dataset = EqtlDataset(val_data, tokenizer, target)
    test_dataset = EqtlDataset(test_data, tokenizer, target)


    # Store datasets
    multitask_datasets['train'] = train_dataset
    multitask_datasets[f"{target}_val"] = val_dataset
    multitask_datasets[f"{target}_test"] = test_dataset

    # Track task info
    task_num_classes = {target: train_dataset.num_labels}

    # Determine max sequence length (considering both ref and alt sequences)
    max_ref_len = max(
        max(len(instance['ref_input_ids']) for instance in train_dataset),
        max(len(instance['ref_input_ids']) for instance in val_dataset),
        max(len(instance['ref_input_ids']) for instance in test_dataset)
    )

    max_alt_len = max(
        max(len(instance['alt_input_ids']) for instance in train_dataset),
        max(len(instance['alt_input_ids']) for instance in val_dataset),
        max(len(instance['alt_input_ids']) for instance in test_dataset)
    )

    max_seq_len = max(max_ref_len, max_alt_len)
    print(f"Max sequence length: {max_seq_len}")

    return multitask_datasets, task_num_classes, max_seq_len



class EqtlDataset(Dataset):
    """Dataset for ClinVar variant data with separate tokenization for reference and alternative sequences."""

    def __init__(self, data, tokenizer, task_name):
        super(EqtlDataset, self).__init__()
        self.task_name = task_name
        self.num_labels = 2

        # Process data
        ref_sequences = []
        alt_sequences = []
        labels = []
        for item in data:
            # Extract reference and alternative sequences (drop variant_type)
            ref, alt, label = item['sequence_ref'],  item['sequence_alt'], item['label']
            ref_sequences.append(ref)
            alt_sequences.append(alt)
            labels.append(label)

        # Tokenize reference sequences
        ref_output = tokenizer(
            ref_sequences,
            return_tensors="pt",
            padding="longest",
            max_length=150000,
            truncation=True,
        )

        # Tokenize alternative sequences
        alt_output = tokenizer(
            alt_sequences,
            return_tensors="pt",
            padding="longest",
            max_length=150000,
            truncation=True,
        )

        # Store tokenized sequences and masks separately
        self.ref_input_ids = ref_output["input_ids"]
        self.ref_attention_mask = ref_output.get("attention_mask")

        self.alt_input_ids = alt_output["input_ids"]
        self.alt_attention_mask = alt_output.get("attention_mask")

        self.labels = labels

    def __len__(self):
        return len(self.ref_input_ids)

    def __getitem__(self, i):
        item = {
            "ref_input_ids": self.ref_input_ids[i],
            "alt_input_ids": self.alt_input_ids[i],
            "labels": self.labels[i],
            "task_name": self.task_name
        }

        if self.ref_attention_mask is not None:
            item["ref_attention_mask"] = self.ref_attention_mask[i]

        if self.alt_attention_mask is not None:
            item["alt_attention_mask"] = self.alt_attention_mask[i]

        return item


class OligogenicPairedDataset(Dataset):
    """Dataset for oligogenic prediction with paired variant processing."""

    def __init__(self, data, tokenizer, task_name="OLIDA", max_length=1024):
        """
        Args:
            data: Output from OligogenicDataWrapper.get_data(paired=True)
            tokenizer: Any HuggingFace-compatible tokenizer
            task_name: Task identifier
            max_length: Maximum sequence length
        """
        super().__init__()
        self.task_name = task_name
        self.num_labels = 2

        # Extract sequences from paired data format
        v1_ref_seqs, v1_alt_seqs = [], []
        v2_ref_seqs, v2_alt_seqs = [], []
        labels = []

        for item in data:
            variants, label = item[0], item[1]
            v1_ref_seqs.append(variants['variant1_ref'])
            v1_alt_seqs.append(variants['variant1_alt'])
            v2_ref_seqs.append(variants['variant2_ref'])
            v2_alt_seqs.append(variants['variant2_alt'])
            labels.append(label)

        # Tokenize all 4 sequence types
        v1_ref_out = tokenizer(v1_ref_seqs, return_tensors="pt", padding="longest",
                               max_length=max_length, truncation=True)
        v1_alt_out = tokenizer(v1_alt_seqs, return_tensors="pt", padding="longest",
                               max_length=max_length, truncation=True)
        v2_ref_out = tokenizer(v2_ref_seqs, return_tensors="pt", padding="longest",
                               max_length=max_length, truncation=True)
        v2_alt_out = tokenizer(v2_alt_seqs, return_tensors="pt", padding="longest",
                               max_length=max_length, truncation=True)

        # Store tokenized data
        self.v1_ref_input_ids = v1_ref_out["input_ids"]
        self.v1_ref_attention_mask = v1_ref_out.get("attention_mask")
        self.v1_alt_input_ids = v1_alt_out["input_ids"]
        self.v1_alt_attention_mask = v1_alt_out.get("attention_mask")

        self.v2_ref_input_ids = v2_ref_out["input_ids"]
        self.v2_ref_attention_mask = v2_ref_out.get("attention_mask")
        self.v2_alt_input_ids = v2_alt_out["input_ids"]
        self.v2_alt_attention_mask = v2_alt_out.get("attention_mask")

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {
            "variant1_ref_input_ids": self.v1_ref_input_ids[i],
            "variant1_alt_input_ids": self.v1_alt_input_ids[i],
            "variant2_ref_input_ids": self.v2_ref_input_ids[i],
            "variant2_alt_input_ids": self.v2_alt_input_ids[i],
            "labels": self.labels[i],
            "task_name": self.task_name
        }

        if self.v1_ref_attention_mask is not None:
            item["variant1_ref_attention_mask"] = self.v1_ref_attention_mask[i]
            item["variant1_alt_attention_mask"] = self.v1_alt_attention_mask[i]
            item["variant2_ref_attention_mask"] = self.v2_ref_attention_mask[i]
            item["variant2_alt_attention_mask"] = self.v2_alt_attention_mask[i]

        return item


def return_maves_dataset(
    tokenizer,
    target="score",
    seq_length=1024,
    val_split=0.1,
    test_split=0.1,
    seed=42,
    all_records=True,
    num_records=2000,
    # Essential filters for training stability
    filter_genes=None,
    experimental_methods=None,
    region_type='all',
    variant_types=None,
    seq_length_range=None,
    max_samples_per_experiment=None,
    # Score normalization
    normalize_scores=False
):
    """
    Load MAVES dataset for variant effect regression with essential filtering.

    Args:
        filter_genes: List of gene names to filter (e.g., ['BRCA1', 'TP53'])
        experimental_methods: List of methods (e.g., ['DMS-BarSeq', 'DMS-TileSeq', 'Enrich2'])
        region_type: 'coding', 'non-coding', or 'all' (default)
        variant_types: List of variant types (e.g., ['sub', 'del', 'ins'])
        seq_length_range: Tuple (min_len, max_len) for sequence length filtering
        max_samples_per_experiment: Maximum samples to take per experiment (None for no limit)
    """
    # Load MAVES data using the data wrapper with essential filters
    mave_wrapper = MAVEDataWrapper(
        num_records=num_records,
        all_records=all_records,
        filter_genes=filter_genes,
        experimental_methods=experimental_methods,
        region_type=region_type,
        variant_types=variant_types,
        seq_length_range=seq_length_range
    )
    raw_data = mave_wrapper.get_data(Seq_length=seq_length, target=target)
    print(f"Loaded {len(raw_data)} MAVES samples for {target} prediction")

    # Group by experiment and limit samples per experiment
    from collections import defaultdict
    exp_data = defaultdict(list)
    skipped = 0

    for seq_pair, score in raw_data:
        try:
            score_float = float(score)
            if np.isnan(score_float) or np.isinf(score_float):
                skipped += 1
                continue
            # Extract experiment ID from annotation
            ref, alt, annotation = seq_pair
            exp_id = annotation.split(',')[0] if annotation else "unknown"
            exp_data[exp_id].append([seq_pair, score_float])
        except (TypeError, ValueError):
            skipped += 1
            continue

    if skipped > 0:
        print(f"Skipped {skipped} samples with invalid scores")

    # Apply per-experiment limits and collect data
    labeled_data = []
    experiment_stats = {}

    # Set random seed once for all experiment sampling
    random.seed(seed)

    for exp_id, samples in exp_data.items():
        original_count = len(samples)

        if max_samples_per_experiment and len(samples) > max_samples_per_experiment:
            samples = random.sample(samples, max_samples_per_experiment)
            used_count = max_samples_per_experiment
        else:
            used_count = original_count

        labeled_data.extend(samples)
        experiment_stats[exp_id] = {'original': original_count, 'used': used_count}


    print(f"\nExperiment sampling summary:")
    print(f"  Total experiments: {len(experiment_stats)}")
    if max_samples_per_experiment:
        print(f"  Max samples per experiment: {max_samples_per_experiment}")
        large_experiments = sum(1 for stats in experiment_stats.values() if stats['original'] > max_samples_per_experiment)
        if large_experiments > 0:
            print(f"  Experiments with sampling applied: {large_experiments}")

    # Shuffle to mix experiments
    random.seed(seed + 1)
    random.shuffle(labeled_data)

    # Calculate score statistics for normalization
    score_mean = 0.0
    score_std = 1.0
    if labeled_data:
        scores = np.array([s[1] for s in labeled_data])
        score_mean = scores.mean()
        score_std = scores.std()

        print(f"\nMAVES Score Statistics:")
        print(f"  Mean: {score_mean:.4f} ± {score_std:.4f}")
        print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"  Data shuffled to mix experiments across batches")

        if normalize_scores:
            print(f"  Score normalization: ENABLED (mean={score_mean:.4f}, std={score_std:.4f})")
        else:
            print(f"  Score normalization: DISABLED")

    print(f"Using {len(labeled_data)} samples for training")

    total = len(labeled_data)
    test_sz = int(total * test_split)
    val_sz = int((total - test_sz) * val_split)
    train_sz = total - test_sz - val_sz


    train_data = labeled_data[:train_sz]
    val_data = labeled_data[train_sz : train_sz + val_sz]
    test_data = labeled_data[train_sz + val_sz :]

    print(f"MAVES data → Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create datasets without normalization
    multitask_datasets = {}
    task_name = f"MAVES_{target}"

    if len(train_data) > 0:
        multitask_datasets["train"] = MAVESDataset(
            train_data, tokenizer, task_name, seq_length,
            normalize=normalize_scores
        )
    if len(val_data) > 0:
        multitask_datasets[f"{task_name}_val"] = MAVESDataset(
            val_data, tokenizer, task_name, seq_length, normalize=normalize_scores
        )
    if len(test_data) > 0:
        multitask_datasets[f"{task_name}_test"] = MAVESDataset(
            test_data, tokenizer, task_name, seq_length, normalize=normalize_scores
        )

    # For regression, output size is 1
    task_num_classes = {task_name: 1}
    max_seq_len = seq_length

    print(f"Max sequence length: {max_seq_len}")

    return multitask_datasets, task_num_classes, max_seq_len


class MAVESDataset(Dataset):
    """Dataset for MAVES variant data with regression targets."""

    def __init__(self, data, tokenizer, task_name, seq_length=1024, normalize=False):
        super(MAVESDataset, self).__init__()
        self.task_name = task_name
        self.num_labels = 1
        self.normalize = normalize

        ref_sequences = []
        alt_sequences = []
        scores = []

        for item in data:
            ref, alt, annotation = item[0]
            ref_sequences.append(ref)
            alt_sequences.append(alt)
            scores.append(float(item[1]))

        # Calculate normalization parameters and apply if requested
        if self.normalize and len(scores) > 1:
            scores_array = np.array(scores)
            score_mean = scores_array.mean()
            score_std = scores_array.std()
            if score_std > 0:
                scores = [(s - score_mean) / score_std for s in scores]
            else:
                print("Warning: Score std is 0, skipping normalization")
                self.normalize = False

        # Tokenize reference sequences
        ref_output = tokenizer(
            ref_sequences,
            return_tensors="pt",
            padding="longest",
            max_length=seq_length,
            truncation=True,
        )

        # Tokenize alternative sequences
        alt_output = tokenizer(
            alt_sequences,
            return_tensors="pt",
            padding="longest",
            max_length=seq_length,
            truncation=True,
        )

        # Store tokenized sequences and masks separately
        self.ref_input_ids = ref_output["input_ids"]
        self.ref_attention_mask = ref_output.get("attention_mask")

        self.alt_input_ids = alt_output["input_ids"]
        self.alt_attention_mask = alt_output.get("attention_mask")

        self.labels = scores

    def __len__(self):
        return len(self.ref_input_ids)

    def __getitem__(self, i):
        item = {
            "ref_input_ids": self.ref_input_ids[i],
            "alt_input_ids": self.alt_input_ids[i],
            "labels": self.labels[i],
            "task_name": self.task_name
        }

        if self.ref_attention_mask is not None:
            item["ref_attention_mask"] = self.ref_attention_mask[i]

        if self.alt_attention_mask is not None:
            item["alt_attention_mask"] = self.alt_attention_mask[i]

        return item


# =============================================================================
# CardioBoost Dataset Functions
# =============================================================================

def return_cardioboost_dataset(
    tokenizer: PreTrainedTokenizer,
    data_dir: str,
    disease_type: str = 'cm',  # 'cm' (cardiomyopathy) or 'arm' (arrhythmia)
    seq_length: int = 1024,  # Default 1024 to match CGC/SMART
    seed: int = 42,
    include_vus: bool = False,  # Whether to include VUS data in training
):
    """
    Load CardioBoost cardiac disease variant dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        data_dir: Path to CardioBoost data directory
        disease_type: 'cm' for cardiomyopathy or 'arm' for arrhythmia
        seq_length: Sequence length (512, 1024, 2048, or 4096)
        seed: Random seed for reproducibility
        include_vus: Whether to include VUS (variants of uncertain significance) in training

    Returns:
        Tuple of (datasets dict, task_num_classes dict, max_seq_len)
    """
    import pandas as pd
    import os

    tokenizer.model_max_length = seq_length

    # Construct file paths
    train_file = os.path.join(data_dir, f"{disease_type}_train_hg19_dna_{seq_length}.csv")
    test_file = os.path.join(data_dir, f"{disease_type}_test_hg19_dna_{seq_length}.csv")

    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    print(f"CardioBoost {disease_type.upper()} Dataset")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")

    # Optionally include VUS data
    if include_vus:
        vus_benign_file = os.path.join(data_dir, f"{disease_type}_vus_benign_hg19_dna_{seq_length}.csv")
        vus_pathogenic_file = os.path.join(data_dir, f"{disease_type}_vus_pathogenic_hg19_dna_{seq_length}.csv")

        if os.path.exists(vus_benign_file) and os.path.exists(vus_pathogenic_file):
            vus_benign_df = pd.read_csv(vus_benign_file)
            vus_pathogenic_df = pd.read_csv(vus_pathogenic_file)

            # Add VUS to training data
            train_df = pd.concat([train_df, vus_benign_df, vus_pathogenic_df], ignore_index=True)
            print(f"  Added VUS benign: {len(vus_benign_df)}, VUS pathogenic: {len(vus_pathogenic_df)}")
            print(f"  Total train samples: {len(train_df)}")

    # Convert to list format expected by ClinVarDataset
    # Format: [[((seq_a, seq_b, variant_type), label], ...]
    def df_to_data_list(df):
        data = []
        for _, row in df.iterrows():
            seq_pair = (row['seq_a'], row['seq_b'], 'SNV')  # variant_type placeholder
            label = int(row['labels'])
            data.append([seq_pair, label])
        return data

    train_data = df_to_data_list(train_df)
    test_data = df_to_data_list(test_df)

    # Split training data into train/val
    random.seed(seed)
    random.shuffle(train_data)

    val_split = 0.15
    val_size = int(len(train_data) * val_split)
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]

    print(f"  After split → Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Print label distribution
    train_labels = [item[1] for item in train_data]
    val_labels = [item[1] for item in val_data]
    test_labels = [item[1] for item in test_data]

    print(f"  Train distribution: benign={train_labels.count(0)}, pathogenic={train_labels.count(1)}")
    print(f"  Val distribution: benign={val_labels.count(0)}, pathogenic={val_labels.count(1)}")
    print(f"  Test distribution: benign={test_labels.count(0)}, pathogenic={test_labels.count(1)}")

    # Create label mapping
    label_to_id = {0: 0, 1: 1}
    task_name = f"cardioboost_{disease_type.upper()}"

    # Create datasets
    train_ds = ClinVarDataset(train_data, tokenizer, task_name, label_to_id)
    val_ds = ClinVarDataset(val_data, tokenizer, task_name, label_to_id)
    test_ds = ClinVarDataset(test_data, tokenizer, task_name, label_to_id)

    # Store datasets
    multitask_datasets = {
        'train': train_ds,
        f"{task_name}_val": val_ds,
        f"{task_name}_test": test_ds,
    }

    task_num_classes = {task_name: 2}  # Binary classification

    return multitask_datasets, task_num_classes, seq_length
