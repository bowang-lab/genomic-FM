#!/bin/bash

# Set model name - modify this variable to change which model to install
MODEL_NAME="${1:-default}"  # Takes first argument or defaults to "default"

echo "Setting up environment for model: $MODEL_NAME"
python --version
pwd
ls

# Core dependencies - installed for all models
echo "Installing core dependencies..."
pip install --user kipoi==0.8.6
pip install --user kipoiseq==0.7.1
pip install --user -r requirements.txt
pip install --user opencv-python-headless numpy decord tqdm omegaconf pytz scikit-learn
pip install --user trl==0.13.0 transformers==4.50.3 accelerate==1.6.0 datasets==3.5.0 wandb
pip install --user tiktoken gdown

# Model-specific installations
case "$MODEL_NAME" in
    "dnabert2"|"dnabert-2")
        echo "Installing DNABERT-2 specific packages..."
        # Override transformers version for DNABERT-2 compatibility
        pip install --user transformers==4.29.0
        # Remove triton for DNABERT-2
        pip uninstall -y triton
        pip uninstall -y triton --user
        ;;
    
    "caduceus")
        echo "Installing Caduceus specific packages..."
        pip install --user mamba-ssm
        pip install --user causal-conv1d
        # Flash attention for efficient training
        # pip install flash-attn --no-build-isolation
        ;;
    
    "mamba"|"mamba-ssm")
        echo "Installing Mamba SSM specific packages..."
        pip install --user mamba-ssm
        pip install --user causal-conv1d
        ;;
    
    "olmo")
        echo "Installing OLMo specific packages..."
        pip install --user ai2-olmo
        ;;
    
    "lucaone"|"LucaOne")
        echo "Installing LucaOne specific packages..."
        # LucaOne uses standard transformers with custom code
        pip install --user transformers>=4.30.0
        echo "Note: LucaOne requires trust_remote_code=True when loading"
        ;;
    
    "all")
        echo "Installing all model packages..."
        pip install --user ai2-olmo
        pip install --user mamba-ssm
        pip install --user causal-conv1d
        pip install --user transformers==4.29.0  # This will override the newer version
        pip uninstall -y triton
        pip uninstall -y triton --user
        echo "Warning: Some packages may conflict when installing all models"
        ;;
    
    "default"|*)
        echo "Using default configuration - core packages only"
        # Remove triton by default as it often causes issues
        pip uninstall -y triton
        pip uninstall -y triton --user
        ;;
esac

# Setup PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
echo "Verifying installation..."
which accelerate
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "Setup complete for model: $MODEL_NAME"
