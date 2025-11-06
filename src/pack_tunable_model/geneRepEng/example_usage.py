#!/usr/bin/env python3
"""
Example script demonstrating how to use geneRepEng for genomic control vector creation
"""

import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import numpy as np

# Import the geneRepEng library
from geneRepEng import ControlModel, ControlVector, GenomicDatasetEntry
from geneRepEng.dataset import create_synthetic_control_dataset, create_clinvar_control_dataset
from geneRepEng.util.omni_dna import create_omni_dna_control_model


def load_omni_dna_model(model_name_or_path: str = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"):
    """
    Load an Omni-DNA or similar genomic model

    Args:
        model_name_or_path: Path or name of the model to load

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

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
    from geneRepEng.extractor import extract_layer_representations

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


if __name__ == "__main__":
    print("Genomic Representation Engineering (geneRepEng) Examples")
    print("=" * 60)

    # Run examples
    try:
        # Main training example
        controlled_model, control_vector = train_control_vector_example()

        # Test different strengths
        test_different_control_strengths()

        # Load saved control vector
        load_and_test_saved_control_vector()

        # Compare representations
        compare_ref_alt_representations()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
