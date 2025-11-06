#!/usr/bin/env python3
"""
Test script for the genomic control vector sweep module.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to sys.path to allow imports
current_dir = Path(__file__).parent
geneRepEng_dir = current_dir.parent
src_dir = geneRepEng_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(geneRepEng_dir))
sys.path.insert(0, str(current_dir))

def test_sweep_imports():
    """Test that all sweep components can be imported."""
    print("Testing sweep imports...")

    try:
        # Test main sweep imports
        import sweep as sweep_module
        from sweep import (
            _load_once, precompute_shared_data, run_single_experiment,
            calculate_classification_metrics, evaluate_genomic_model
        )
        print("✓ Main sweep imports successful")

        # Test geneRepEng imports
        from geneRepEng import ControlModel, ControlVector, GenomicDatasetEntry
        from geneRepEng.dataset import create_synthetic_control_dataset
        # Note: omni_dna import is commented out in __init__.py, so we skip it for now
        print("✓ geneRepEng imports successful")

        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_synthetic_dataset():
    """Test synthetic dataset creation."""
    print("Testing synthetic dataset creation...")

    try:
        from geneRepEng.dataset import create_synthetic_control_dataset

        dataset = create_synthetic_control_dataset(
            n_samples=10,
            seq_length=50,
            mutation_rate=0.02,
            seed=42
        )

        assert len(dataset) == 10
        assert all(len(entry.ref_sequence) == 50 for entry in dataset.entries)
        assert all(len(entry.alt_sequence) == 50 for entry in dataset.entries)

        print("✓ Synthetic dataset creation successful")
        return True
    except Exception as e:
        print(f"✗ Synthetic dataset test failed: {e}")
        return False

def test_classification_metrics():
    """Test classification metrics calculation."""
    print("Testing classification metrics...")

    try:
        import numpy as np
        from sweep import calculate_classification_metrics

        # Create test data
        predictions = np.array([0, 1, 1, 0, 1])
        labels = np.array([0, 1, 0, 0, 1])

        metrics = calculate_classification_metrics(predictions, labels)

        expected_keys = ["accuracy", "f1", "matthews_correlation", "precision", "recall"]
        assert all(key in metrics for key in expected_keys)
        assert all(isinstance(metrics[key], float) for key in expected_keys)

        print("✓ Classification metrics test successful")
        return True
    except Exception as e:
        print(f"✗ Classification metrics test failed: {e}")
        return False

def test_model_loading():
    """Test model loading functionality (without actually loading large models)."""
    print("Testing model loading structure...")

    try:
        from sweep import _load_once

        # Test that the function exists and has correct signature
        import inspect
        sig = inspect.signature(_load_once)
        expected_params = ['model_id', 'max_length']

        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params, f"Expected {expected_params}, got {actual_params}"

        print("✓ Model loading structure test successful")
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def test_config_files():
    """Test that configuration files exist and are readable."""
    print("Testing configuration files...")

    try:
        sweep_dir = Path(__file__).parent

        # Check required files exist
        required_files = [
            "sweep.py",
            "run_sweep.py",
            "analyze_results.py",
            "config.md",
            "README.md"
        ]

        for file_name in required_files:
            file_path = sweep_dir / file_name
            assert file_path.exists(), f"Missing file: {file_name}"
            assert file_path.stat().st_size > 0, f"Empty file: {file_name}"

        print("✓ Configuration files test successful")
        return True
    except Exception as e:
        print(f"✗ Configuration files test failed: {e}")
        return False

def test_argument_parsing():
    """Test argument parsing for sweep scripts."""
    print("Testing argument parsing...")

    try:
        # Test sweep.py arguments
        sys.argv = ['sweep.py', '--help']

        # We can't actually run argparse with --help as it exits
        # So we'll just test the module import
        import sweep
        assert hasattr(sweep, 'parse_args')

        # Test run_sweep.py
        import run_sweep
        assert hasattr(run_sweep, 'main')

        print("✓ Argument parsing test successful")
        return True
    except Exception as e:
        print(f"✗ Argument parsing test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("Running Genomic Control Vector Sweep Tests")
    print("=" * 50)

    tests = [
        test_sweep_imports,
        test_synthetic_dataset,
        test_classification_metrics,
        test_model_loading,
        test_config_files,
        test_argument_parsing,
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

    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All sweep tests passed!")
        return True
    else:
        print("❌ Some sweep tests failed.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
