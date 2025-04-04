import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from transformers import PreTrainedTokenizer
from datasets import load_dataset
import logging
import typing
from typing import Dict, Sequence
from ..dataloader.data_wrapper import ClinVarDataWrapper

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

def return_clinvar_multitask_dataset(tokenizer: PreTrainedTokenizer, target='CLNDN', disease_subset=True,
                                    seq_length=1024, val_split=0.1, test_split=0.1, seed=42):
    """
    Load ClinVar datasets for multi-task learning.

    Args:
        tokenizer: PreTrainedTokenizer for tokenizing sequences
        target: Target feature to predict ('CLNDN' for disease names by default, or 'CLNSIG' for clinical significance)
        disease_subset: Whether to use a disease subset
        seq_length: Sequence length for extraction
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility

    Returns:
        tuple: (multitask_datasets, task_num_classes, max_seq_len)
    """
    import torch
    from torch.utils.data import Dataset, Subset
    import numpy as np
    from tqdm import tqdm
    import random

    # Get ClinVar data
    clinvar_wrapper = ClinVarDataWrapper(all_records=True)
    data = clinvar_wrapper.get_data(Seq_length=seq_length, target=target, disease_subset=disease_subset)
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
    elif target == 'CLNSIG':
        task_name = 'CLNSIG'
        all_labels = ['Benign', 'Likely_benign', 'Likely_pathogenic', 'Pathogenic']
        label_to_id = {label: idx for idx, label in enumerate(all_labels)}
        num_labels = len(all_labels)
        print(f"Using {num_labels} pathogenicity classes")
    else:
        raise ValueError(f"Unsupported target: {target}")
    # Create datasets
    train_dataset = ClinVarDataset(train_data, tokenizer, task_name, label_to_id)
    val_dataset = ClinVarDataset(val_data, tokenizer, task_name, label_to_id)
    test_dataset = ClinVarDataset(test_data, tokenizer, task_name, label_to_id)

    # Store datasets
    multitask_datasets['train'] = train_dataset
    multitask_datasets[f"{task_name}_val"] = val_dataset
    multitask_datasets[f"{task_name}_test"] = test_dataset

    # Track task info
    task_num_classes = {task_name: num_labels}

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

class ClinVarDataset(Dataset):
    """Dataset for ClinVar variant data with separate tokenization for reference and alternative sequences."""

    def __init__(self, data, tokenizer, task_name, label_to_id):
        super(ClinVarDataset, self).__init__()
        self.task_name = task_name
        self.num_labels = len(label_to_id)

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
