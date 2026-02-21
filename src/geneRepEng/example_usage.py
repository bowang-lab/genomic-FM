#!/usr/bin/env python3
"""
Example script demonstrating how to use geneRepEng for genomic control vector creation
"""

import os
import json
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM
from typing import List, Dict
import numpy as np

# Import the geneRepEng library
from src.geneRepEng import ControlModel, ControlVector, GenomicDatasetEntry
from src.geneRepEng.dataset import (
    create_synthetic_control_dataset,
    create_clinvar_control_dataset,
    load_cgc_primary_findings,
    load_cardiac_benign_variants,
    create_balanced_control_dataset
)
from src.geneRepEng.util.omni_dna import create_omni_dna_control_model


# Model registry for determining base models from checkpoint paths
MODEL_REGISTRY = {
    "nt": {
        "identifier": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "local_path": "root/models/nt",
        "patterns": ["_nt_", "/nt/", "-nt-", "nucleotide_transformer", "nucleotide-transformer"],
        "model_class": AutoModelForMaskedLM,
    },
    "omni_dna_116m": {
        "identifier": "zehui127/Omni-DNA-116M",
        "local_path": "root/models/omni_dna_116m",
        "patterns": ["omni_dna_116m", "omni-dna", "omni_dna", "omnidna"],
        "model_class": AutoModelForCausalLM,
    },
    "dnabert2": {
        "identifier": "zhihan1996/DNABERT-2-117M",
        "local_path": "root/models/dnabert2",
        "patterns": ["dnabert2", "dnabert-2", "dnabert"],
        "model_class": AutoModel,
    },
    "hyenadna": {
        "identifier": "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "local_path": "root/models/hyenadna",
        "patterns": ["hyenadna", "hyena-dna", "hyena"],
        "model_class": AutoModel,
    },
    "caduceus": {
        "identifier": "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16",
        "local_path": "root/models/caduceus",
        "patterns": ["caduceus"],
        "model_class": AutoModel,
    },
    "gena-lm": {
        "identifier": "AIRI-Institute/gena-lm-bert-base-t2t",
        "local_path": "root/models/gena-lm",
        "patterns": ["gena-lm", "gena_lm", "genalm"],
        "model_class": AutoModel,
    },
    "gpn-star": {
        "identifier": "songlab/gpn-star-hg38-v100-200m",
        "local_path": "root/models/gpn-star-hg38-v100-200m",
        "patterns": ["gpn-star", "gpn_star", "gpnstar"],
        "model_class": AutoModelForMaskedLM,
    },
}


def determine_base_model_from_path(model_path: str) -> tuple:
    """
    Determine base model identifier from checkpoint path.

    Args:
        model_path: Path to checkpoint

    Returns:
        Tuple of (model_key, identifier, local_path, model_class)
    """
    path_lower = str(model_path).lower()

    # Check patterns in order (more specific first)
    for model_key, config in MODEL_REGISTRY.items():
        for pattern in config["patterns"]:
            if pattern in path_lower:
                return model_key, config["identifier"], config["local_path"], config["model_class"]

    # Default to omni_dna_116m
    default = MODEL_REGISTRY["omni_dna_116m"]
    return "omni_dna_116m", default["identifier"], default["local_path"], default["model_class"]


def find_best_checkpoint(model_dir: Path) -> Path:
    """
    Find best checkpoint from training run, or use base model if no checkpoints.

    Args:
        model_dir: Directory containing checkpoint subdirectories or base model

    Returns:
        Path to best checkpoint directory, or model_dir if base model
    """
    model_dir = Path(model_dir)

    checkpoint_dirs = sorted(
        [d for d in model_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: int(x.name.split('-')[1])
    )

    if checkpoint_dirs:
        latest_checkpoint = checkpoint_dirs[-1]
        trainer_state_path = latest_checkpoint / "trainer_state.json"

        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)

            best_model_path = state.get('best_model_checkpoint')
            best_metric = state.get('best_metric')

            if best_model_path:
                best_checkpoint = Path(best_model_path)
                if best_checkpoint.is_absolute():
                    best_checkpoint_dir = best_checkpoint
                else:
                    best_checkpoint_dir = model_dir / best_checkpoint.name

                if best_checkpoint_dir.exists():
                    metric_str = f" (metric={best_metric:.4f})" if best_metric is not None else ""
                    print(f"Using best checkpoint: {best_checkpoint_dir.name}{metric_str}")
                    return best_checkpoint_dir

        print(f"Using latest checkpoint: {latest_checkpoint.name}")
        return latest_checkpoint

    print(f"No checkpoints found in {model_dir.name}, using base model")
    return model_dir


def load_omni_dna_model(model_name_or_path: str = None):
    """
    Load an Omni-DNA or similar genomic model

    Args:
        model_name_or_path: Path or name of the model to load.
                           If None, uses local model or zehui127/Omni-DNA-116M
                           Can be a checkpoint path like 'smart_pretrain_model_nt_CLNSIG_...'

    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = None
    base_model_path = None
    model_class = AutoModelForCausalLM  # Default

    # Match training code logic
    if model_name_or_path is None:
        local_model_path = "./root/models/omni_dna_116m"
        if os.path.exists(local_model_path):
            base_model_path = local_model_path
            print(f"Using local model from {base_model_path}")
        else:
            base_model_path = "zehui127/Omni-DNA-116M"
            print(f"Using HuggingFace model: {base_model_path}")
    else:
        # Check if it's a relative path to root/models
        if not os.path.exists(model_name_or_path) and os.path.exists(f"./root/models/{model_name_or_path}"):
            model_name_or_path = f"./root/models/{model_name_or_path}"
        print(f"Loading model: {model_name_or_path}")

        # Check if this is a checkpoint directory (has checkpoint-* subdirs or is a checkpoint itself)
        if os.path.isdir(model_name_or_path):
            resolved_path = find_best_checkpoint(Path(model_name_or_path))

            # Check if the resolved path has a config.json with model_type
            config_path = resolved_path / "config.json"
            has_valid_config = False
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                has_valid_config = 'model_type' in config

            if has_valid_config:
                # Can load directly from this path
                base_model_path = str(resolved_path)
                # Still need to determine model class from path
                model_key, _, _, model_class = determine_base_model_from_path(model_name_or_path)
            else:
                # Need to load base model and apply checkpoint weights
                checkpoint_path = resolved_path
                model_key, identifier, local_path, model_class = determine_base_model_from_path(model_name_or_path)
                print(f"Detected base model: {model_key}")

                # Try local path first, then HuggingFace
                if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
                    base_model_path = local_path
                    print(f"Using local base model: {base_model_path}")
                else:
                    base_model_path = identifier
                    print(f"Using HuggingFace base model: {base_model_path}")
        else:
            base_model_path = model_name_or_path
            # Determine model class from path
            _, _, _, model_class = determine_base_model_from_path(model_name_or_path)

    # Load tokenizer and model from base model
    print(f"Loading with model class: {model_class.__name__}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = model_class.from_pretrained(base_model_path, trust_remote_code=True)

    # Load checkpoint weights if we have a checkpoint
    if checkpoint_path is not None:
        checkpoint_file = checkpoint_path / "pytorch_model.bin"
        if not checkpoint_file.exists():
            # Try safetensors format
            checkpoint_file = checkpoint_path / "model.safetensors"

        if checkpoint_file.exists():
            print(f"Loading checkpoint weights from: {checkpoint_file.name}")
            if checkpoint_file.suffix == '.bin':
                state_dict = torch.load(checkpoint_file, map_location='cpu')
            else:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_file)

            # Filter and load weights that match the model
            model_state = model.state_dict()
            filtered_state = {}
            for key, value in state_dict.items():
                # Handle wrapped model prefixes
                clean_key = key.replace('module.', '').replace('base_model.', '')
                if clean_key in model_state and model_state[clean_key].shape == value.shape:
                    filtered_state[clean_key] = value

            if filtered_state:
                model.load_state_dict(filtered_state, strict=False)
                print(f"Loaded {len(filtered_state)}/{len(model_state)} weights from checkpoint")
            else:
                print("Warning: No matching weights found in checkpoint")
        else:
            print(f"Warning: No checkpoint weights found at {checkpoint_path}")

    # Set appropriate sequence length
    if hasattr(tokenizer, 'model_max_length'):
        if tokenizer.model_max_length > 100000:  # If unreasonably large
            tokenizer.model_max_length = 1024
    else:
        tokenizer.model_max_length = 1024

    print(f"Model loaded successfully. Max sequence length: {tokenizer.model_max_length}")
    return model, tokenizer


def create_example_genomic_dataset(n_samples: int = 100) -> List[GenomicDatasetEntry]:
    """
    Create an example dataset with genomic sequence pairs

    Args:
        n_samples: Number of sequence pairs to create

    Returns:
        List of GenomicDatasetEntry objects
    """
    print(f"Creating synthetic dataset with {n_samples} samples...")

    # Use the synthetic dataset creation function
    dataset = create_synthetic_control_dataset(
        n_samples=n_samples,
        seq_length=100,
        mutation_rate=0.05,
        seed=42
    )

    return dataset.entries


def train_control_vector_example():
    """
    Example of training a control vector using genomic data
    """
    print("=== Genomic Control Vector Training Example ===\n")

    # Step 1: Load the genomic model
    model, tokenizer = load_omni_dna_model()

    # Step 2: Create or load genomic dataset
    dataset_entries = create_example_genomic_dataset(n_samples=50)

    # Step 3: Train control vector
    print("Training control vector...")
    control_vector = ControlVector.train(
        model=model,
        processors=[tokenizer],
        dataset=dataset_entries,
        max_batch_size=8,
        method="pca_diff"
    )

    print(f"Control vector trained for {len(control_vector.directions)} layers")

    # Step 4: Create controlled model
    print("Creating controlled model...")
    controlled_model = create_omni_dna_control_model(
        model=model,
        layer_ids=list(control_vector.directions.keys())
    )

    # Step 5: Apply control vector
    controlled_model.set_control(control_vector, strength=1.0)

    print("Control vector applied successfully!")

    # Step 6: Test the controlled model
    test_sequence = "ATGCGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    print(f"\nTesting with sequence: {test_sequence}")

    # Tokenize test sequence
    inputs = tokenizer(
        test_sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    )

    # Get original model output
    with torch.no_grad():
        original_output = model(**inputs, output_hidden_states=True)
        original_hidden = original_output.hidden_states[-1][:, -1, :]  # Last layer, last token

    # Get controlled model output
    with torch.no_grad():
        controlled_output = controlled_model(**inputs, output_hidden_states=True)
        controlled_hidden = controlled_output.hidden_states[-1][:, -1, :]  # Last layer, last token

    # Compare outputs
    difference = torch.norm(controlled_hidden - original_hidden).item()
    print(f"Difference in final hidden states: {difference:.4f}")

    # Step 7: Save control vector
    control_vector.save("genomic_control_vector.npz")
    print("Control vector saved to: genomic_control_vector.npz")

    return controlled_model, control_vector


def test_different_control_strengths():
    """
    Example of testing different control vector strengths
    """
    print("\n=== Testing Different Control Strengths ===\n")

    # Load model and create control vector (simplified)
    model, tokenizer = load_omni_dna_model()
    dataset_entries = create_example_genomic_dataset(n_samples=20)

    control_vector = ControlVector.train(
        model=model,
        processors=[tokenizer],
        dataset=dataset_entries,
        max_batch_size=4,
        method="mean_diff"  # Use simpler method for this example
    )

    controlled_model = create_omni_dna_control_model(model=model)

    # Test sequence
    test_sequence = "ATGCGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    inputs = tokenizer(test_sequence, return_tensors="pt", padding=True, truncation=True)

    # Test different control strengths
    strengths = [0.0, 0.5, 1.0, 2.0]

    with torch.no_grad():
        original_output = model(**inputs, output_hidden_states=True)
        original_hidden = original_output.hidden_states[-1][:, -1, :]

    print(f"Testing control strengths: {strengths}")

    for strength in strengths:
        controlled_model.set_control(control_vector, strength=strength)

        with torch.no_grad():
            controlled_output = controlled_model(**inputs, output_hidden_states=True)
            controlled_hidden = controlled_output.hidden_states[-1][:, -1, :]

        difference = torch.norm(controlled_hidden - original_hidden).item()
        print(f"Strength {strength}: Difference = {difference:.4f}")

    controlled_model.clear_control()
    print("Control cleared.")


def load_and_test_saved_control_vector():
    """
    Example of loading a previously saved control vector
    """
    print("\n=== Loading Saved Control Vector ===\n")

    try:
        # Load the saved control vector
        control_vector = ControlVector.load("genomic_control_vector.npz")
        print(f"Loaded control vector with {len(control_vector.directions)} layers")

        # Load model
        model, tokenizer = load_omni_dna_model()

        # Create controlled model
        controlled_model = create_omni_dna_control_model(model=model)
        controlled_model.set_control(control_vector, strength=1.0)

        print("Successfully loaded and applied saved control vector!")

    except FileNotFoundError:
        print("No saved control vector found. Run train_control_vector_example() first.")


def compare_ref_alt_representations():
    """
    Example of comparing reference vs alternative sequence representations
    """
    print("\n=== Comparing Reference vs Alternative Representations ===\n")

    model, tokenizer = load_omni_dna_model()

    # Create example sequence pairs
    ref_sequences = [
        "ATGCGTACGTACGTACGTACGTACGTAC",
        "GCTAGCTAGCTAGCTAGCTAGCTAGCT",
        "TTAACCGGTTAACCGGTTAACCGGTTA"
    ]

    alt_sequences = [
        "ATGCGTACGTACGTACGTACGTACGAC",  # One mutation
        "GCTAGCTAGCTAGCTAGCTAGCTAGTT",  # One mutation
        "TTAACCGGTTAACCGGTTAACCGGCTA"   # One mutation
    ]

    print("Analyzing sequence pairs:")
    for i, (ref, alt) in enumerate(zip(ref_sequences, alt_sequences)):
        print(f"  Pair {i+1}:")
        print(f"    Ref: {ref}")
        print(f"    Alt: {alt}")

    # Get representations
    from src.geneRepEng.extractor import extract_layer_representations

    ref_reps = extract_layer_representations(
        model, tokenizer, ref_sequences, layer_ids=[6, 12, 18], batch_size=3
    )

    alt_reps = extract_layer_representations(
        model, tokenizer, alt_sequences, layer_ids=[6, 12, 18], batch_size=3
    )

    # Analyze differences
    for layer_id in ref_reps.keys():
        ref_layer = ref_reps[layer_id]
        alt_layer = alt_reps[layer_id]

        differences = alt_layer - ref_layer
        mean_diff = np.mean(np.linalg.norm(differences, axis=1))

        print(f"Layer {layer_id}: Mean representation difference = {mean_diff:.4f}")


def train_cgc_cardiac_control_vector(model_path: str = None):
    """
    Train control vector using CGC pediatric cardiac patient variants.

    Uses real pathogenic variants from CGC patients as the "alternative" and
    benign ClinVar cardiac variants as the "reference" to learn pathogenicity direction.

    Args:
        model_path: Path to the model to use. Can be:
                   - Full path: "root/models/pretrain_model_omni_dna_116m_CLNDN"
                   - Model name: "pretrain_model_omni_dna_116m_CLNDN" (will look in root/models/)
                   - None: Uses default omni_dna_116m
    """
    print("\n=== CGC Cardiac Pathogenicity Control Vector Training ===\n")

    # Step 1: Load the genomic model
    model, tokenizer = load_omni_dna_model(model_path)

    # Step 2: Load CGC pathogenic variants (primary findings)
    print("Loading CGC primary findings (pathogenic variants)...")
    cgc_pathogenic = load_cgc_primary_findings(
        csv_path="root/data/primary_findings_analysis/primary_findings_analysis_results.csv",
        genome_fa="root/data/hg19.fa",
        seq_length=1024
    )
    print(f"Loaded {len(cgc_pathogenic)} CGC pathogenic variants")

    # Step 3: Load ClinVar benign cardiac variants as controls
    print("\nLoading ClinVar benign cardiac variants (controls)...")
    clinvar_benign = load_cardiac_benign_variants(
        n_samples=500,  # Use 500 benign variants
        seq_length=1024,
        seed=42
    )
    print(f"Loaded {len(clinvar_benign)} ClinVar benign variants")

    # Step 4: Create balanced dataset
    print("\nCreating balanced control dataset...")
    balanced_dataset = create_balanced_control_dataset(
        pathogenic_dataset=cgc_pathogenic,
        benign_dataset=clinvar_benign,
        balance_method="upsample",  # Upsample pathogenic to match benign count
        seed=42
    )

    # Step 5: Train control vector
    print("\nTraining control vector...")
    control_vector = ControlVector.train(
        model=model,
        processors=[tokenizer],
        dataset=balanced_dataset.entries,
        max_batch_size=4,  # Small batch size for large sequences
        method="pca_diff"
    )

    print(f"Control vector trained for {len(control_vector.directions)} layers")

    # Step 6: Create controlled model
    print("\nCreating controlled model...")
    controlled_model = create_omni_dna_control_model(
        model=model,
        layer_ids=list(control_vector.directions.keys())
    )

    # Step 7: Apply control vector
    controlled_model.set_control(control_vector, strength=1.0)
    print("Control vector applied successfully!")

    # Step 8: Test on a CGC variant
    if len(cgc_pathogenic) > 0:
        test_entry = cgc_pathogenic.entries[0]
        test_sequence = test_entry.alt_sequence  # Use pathogenic alt sequence

        print(f"\nTesting with CGC pathogenic variant...")
        print(f"Sequence length: {len(test_sequence)}")

        # Tokenize test sequence
        inputs = tokenizer(
            test_sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        )

        # Get original model output
        with torch.no_grad():
            original_output = model(**inputs, output_hidden_states=True)
            original_hidden = original_output.hidden_states[-1][:, -1, :]

        # Get controlled model output
        with torch.no_grad():
            controlled_output = controlled_model(**inputs, output_hidden_states=True)
            controlled_hidden = controlled_output.hidden_states[-1][:, -1, :]

        # Compare outputs
        difference = torch.norm(controlled_hidden - original_hidden).item()
        print(f"Difference in final hidden states: {difference:.4f}")

    # Step 9: Save control vector with model name in filename
    if model_path:
        # Extract model name from path (e.g., "smart_pretrain_model_nt_CLNSIG_..." -> use as is)
        model_name = Path(model_path).name.replace('/', '_')
    else:
        model_name = "omni_dna_116m"
    output_path = f"root/output/cgc_cardiac_pathogenicity_control_{model_name}.npz"
    control_vector.save(output_path)
    print(f"\nControl vector saved to: {output_path}")

    # Step 10: Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  CGC pathogenic variants: {len(cgc_pathogenic)}")
    print(f"  ClinVar benign variants: {len(clinvar_benign)}")
    print(f"  Total training samples: {len(balanced_dataset)}")
    print(f"  Layers with control: {len(control_vector.directions)}")
    print(f"  Saved to: {output_path}")
    print("=" * 60)

    return controlled_model, control_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genomic Representation Engineering (geneRepEng) Examples"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model in root/models/ (e.g., 'pretrain_model_omni_dna_116m_CLNDN' or full path)"
    )
    args = parser.parse_args()

    print("Genomic Representation Engineering (geneRepEng) Examples")
    print("=" * 60)
    if args.model_path:
        print(f"Using model: {args.model_path}")

    # Run examples
    try:
        # CGC Cardiac Control Vector Training (PRIMARY USE CASE)
        print("\n" + "=" * 60)
        print("Training CGC Cardiac Pathogenicity Control Vector")
        print("=" * 60)
        controlled_model, control_vector = train_cgc_cardiac_control_vector(
            model_path=args.model_path
        )

        print("\n" + "=" * 60)
        print("CGC training completed successfully!")
        print("=" * 60)

        # Optional: Run other examples (commented out by default)
        # test_different_control_strengths()
        # load_and_test_saved_control_vector()
        # compare_ref_alt_representations()

    except Exception as e:
        print(f"\nError running CGC training: {e}")
        import traceback
        traceback.print_exc()
