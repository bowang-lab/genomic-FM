#!/usr/bin/env python3
"""
Basic tests for geneRepEng functionality
"""

import sys
import os
import torch
import numpy as np
from typing import List

# Add the geneRepEng module to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all main components can be imported"""
    print("Testing imports...")

    try:
        from geneRepEng import ControlModel, ControlVector, GenomicDatasetEntry, CustomFunctions
        from geneRepEng.dataset import GenomicDataset, create_synthetic_control_dataset
        from geneRepEng.util.omni_dna import generate_random_dna_sequence, mutate_dna_sequence
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation functionality"""
    print("Testing dataset creation...")

    try:
        # Test synthetic dataset creation
        dataset = create_synthetic_control_dataset(
            n_samples=10,
            seq_length=50,
            mutation_rate=0.02,
            seed=42
        )

        assert len(dataset) == 10
        assert all(isinstance(entry.ref_sequence, str) for entry in dataset.entries)
        assert all(isinstance(entry.alt_sequence, str) for entry in dataset.entries)
        assert all(len(entry.ref_sequence) == 50 for entry in dataset.entries)
        assert all(len(entry.alt_sequence) == 50 for entry in dataset.entries)

        print("✓ Dataset creation successful")
        return True
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False

def test_genomic_dataset_entry():
    """Test GenomicDatasetEntry functionality"""
    print("Testing GenomicDatasetEntry...")

    try:
        from geneRepEng import GenomicDatasetEntry

        entry = GenomicDatasetEntry(
            ref_sequence="ATGCGTACGTAC",
            alt_sequence="ATGCGTACGTCC",
            label=1
        )

        assert entry.ref_sequence == "ATGCGTACGTAC"
        assert entry.alt_sequence == "ATGCGTACGTCC"
        assert entry.label == 1

        print("✓ GenomicDatasetEntry test successful")
        return True
    except Exception as e:
        print(f"✗ GenomicDatasetEntry test failed: {e}")
        return False

def test_dna_utilities():
    """Test DNA utility functions"""
    print("Testing DNA utilities...")

    try:
        from geneRepEng.util.omni_dna import (
            generate_random_dna_sequence,
            mutate_dna_sequence,
            validate_dna_sequence
        )

        # Test sequence generation
        seq = generate_random_dna_sequence(100, seed=42)
        assert len(seq) == 100
        assert validate_dna_sequence(seq)

        # Test mutation
        mutated_seq = mutate_dna_sequence(seq, mutation_rate=0.1, seed=42)
        assert len(mutated_seq) == 100
        assert validate_dna_sequence(mutated_seq)
        assert seq != mutated_seq  # Should be different

        print("✓ DNA utilities test successful")
        return True
    except Exception as e:
        print(f"✗ DNA utilities test failed: {e}")
        return False

def test_control_vector_structure():
    """Test ControlVector data structure"""
    print("Testing ControlVector structure...")

    try:
        from geneRepEng import ControlVector

        # Create a mock control vector
        directions = {
            0: np.random.randn(768),
            1: np.random.randn(768),
            2: np.random.randn(768)
        }

        control_vector = ControlVector(
            model_type="test_model",
            directions=directions
        )

        assert control_vector.model_type == "test_model"
        assert len(control_vector.directions) == 3
        assert all(isinstance(d, np.ndarray) for d in control_vector.directions.values())

        print("✓ ControlVector structure test successful")
        return True
    except Exception as e:
        print(f"✗ ControlVector structure test failed: {e}")
        return False

def test_custom_functions():
    """Test CustomFunctions functionality"""
    print("Testing CustomFunctions...")

    try:
        from geneRepEng import CustomFunctions

        # Test default initialization
        custom_funcs = CustomFunctions()
        assert custom_funcs.F_batched_get_hiddens is not None
        assert custom_funcs.F_batched_preprocess is not None

        print("✓ CustomFunctions test successful")
        return True
    except Exception as e:
        print(f"✗ CustomFunctions test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    print("Running geneRepEng Basic Tests")
    print("=" * 40)

    tests = [
        test_imports,
        test_genomic_dataset_entry,
        test_dataset_creation,
        test_dna_utilities,
        test_control_vector_structure,
        test_custom_functions
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
        print()

    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
