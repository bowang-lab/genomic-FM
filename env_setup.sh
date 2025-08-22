#!/bin/bash

# IMPORTANT: This script should be run INSIDE an activated conda environment
# Usage: conda activate genomic-fm-lucaone && ./env.sh [model_name]

set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Set model name - modify this variable to change which model to install
MODEL_NAME="${1:-default}"  # Takes first argument or defaults to "default"

echo "Setting up environment for model: $MODEL_NAME"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
pwd
ls

# Verify we're in a conda environment
if [[ "$CONDA_DEFAULT_ENV" == "" ]]; then
    echo "ERROR: No conda environment activated!"
    echo "Please run: conda activate your-env-name"
    exit 1
fi

echo "Active conda environment: $CONDA_DEFAULT_ENV"

# Core dependencies - installed to the conda env (NO --user!)
echo "Installing core dependencies..."
pip install kipoi==0.8.6
pip install kipoiseq==0.7.1
pip install -r requirements.txt
pip install opencv-python-headless numpy decord tqdm omegaconf pytz scikit-learn
pip install trl==0.13.0 transformers==4.50.3 accelerate==1.6.0 datasets==3.5.0 wandb


# Model-specific installations
case "$MODEL_NAME" in
    "dnabert2"|"dnabert-2")
        echo "Installing DNABERT-2 specific packages..."
	# pip install --user transformers==4.29.0
	pip install tiktoken gdown tiktoken datasets wandb
        # Remove triton for DNABERT-2
        pip uninstall -y triton || true
        ;;
    
    "caduceus")
        echo "Installing Caduceus specific packages..."
        pip install mamba-ssm
        pip install causal-conv1d
        pip install accelerate
        ;;
    
    "olmo")
        echo "Installing OLMo specific packages..."
        pip install ai2-olmo
        ;;
    
    "lucaone")
        echo "Installing LucaOne specific packages..."
        # LucaOne uses standard transformers with custom code
        pip install lucagplm
        pip install tokenizers==0.19.1
        pip install transformers==4.41.2
        pip install fair-esm
        pip install statsmodels
        ;;
    
esac


# Verify installation
echo "Verifying installation..."
which accelerate

echo "Setup complete for model: $MODEL_NAME"
