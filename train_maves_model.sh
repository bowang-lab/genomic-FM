#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J maves_finetune
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=240G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:2 # match ddp.yaml num_processes
#SBATCH --ntasks=1 # Keep as 1 since we'll use accelerate launch
#SBATCH --output=logs/maves_finetune_output_%j.log
#SBATCH --error=logs/maves_finetune_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vallisubasri@gmail.com

# Exit on any error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

cd /cluster/projects/bwanggroup/vsubasri/genomic-FM

source ~/miniconda3/etc/profile.d/conda.sh

conda activate genomic-fm

# Disable wandb syncing for now
wandb offline

# Training parameters
MODEL="nt"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-msa-sapiens

# Essential MAVES Filtering Options
# Uncomment and customize as needed for training stability

# Gene filtering: Focus on specific genes of interest
#GENE_FILTER="--filter_genes UBE2I,BRCA1,TP53"

# Method filtering: BIOLOGICALLY PRINCIPLED CATEGORIES
# AVAILABLE CATEGORIES (7 distinct biological groupings):
#   🔬 DMS - Deep mutational scanning (all variants, single/multi-readout)
#   📊 REGULATORY_FUNCTION - MPRA, promoter/enhancer activity
#   🤝 BINDING_INTERACTION - Y2H, phage display, autoubiquitination
#   🔬 BIOPHYSICAL_STABILITY - Protease digestion, folding, HPLC
#   📈 ABUNDANCE_EXPRESSION - RNA/protein levels, fluorescence
#   🎯 FITNESS_GROWTH - Organismal fitness, growth, survival
#   ⚠️ COMPUTATIONAL_PROCESSED - Enrich2, model outputs (use with caution)
#
# Using individual methods:
#METHOD_FILTER="--experimental_methods DMS-BarSeq,trypsin digestion"

# Coding region filtering: protein-coding vs non-coding
#CODING_FILTER="--coding_only true"

# Sequence length filtering: computational efficiency and consistency
#LENGTH_FILTER="--seq_length_range 200,800"


# MAVE METHOD CATEGORIES (quality: high → low):

# 🔬 DEEP MUTATIONAL SCANNING (highest quality):
#FILTER_FLAGS="--experimental_methods DMS"

# 🎯 ORGANISMAL FITNESS (clean functional signals):
#FILTER_FLAGS="--experimental_methods FITNESS_GROWTH"

# 🤝 PROTEIN INTERACTIONS (binding/enzymatic):
#FILTER_FLAGS="--experimental_methods BINDING_INTERACTION"

# 📊 REGULATORY ELEMENTS (promoter/enhancer):
#FILTER_FLAGS="--experimental_methods REGULATORY_FUNCTION"

# 🔬 PROTEIN STABILITY (structural properties):
#FILTER_FLAGS="--experimental_methods BIOPHYSICAL_STABILITY"

# 📈 EXPRESSION LEVELS (noisiest):
#FILTER_FLAGS="--experimental_methods ABUNDANCE_EXPRESSION"

# ⚠️ COMPUTATIONAL OUTPUTS (use with caution):
#FILTER_FLAGS="--experimental_methods COMPUTATIONAL"

# ✅ COMPATIBLE COMBINATIONS:
#FILTER_FLAGS="--experimental_methods DMS,FITNESS_GROWTH"

# 🚀 PROGRESSIVE TRAINING (recommended order):
# Stage 1: DMS → Stage 2: FITNESS_GROWTH → Stage 3: BINDING_INTERACTION

FILTER_FLAGS=""  # No filtering by default

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

echo "============================================"
echo "Starting MAVES Regression Training"
echo "Model: $MODEL"
echo "Task: MAVES"
if [ -n "$FILTER_FLAGS" ]; then
    echo "Filters: $FILTER_FLAGS"
fi
echo "============================================"

# Run MAVES regression training
echo "=================================="
echo "Starting MAVES score regression training..."
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer \
    --model "$MODEL" \
    --task MAVES \
    --seed 127 \
    --learning_rate 0.000005 \
    --batch_size 8 \
    --num_epochs 10 \
    --max_grad_norm 1.0 \
    --num_workers 8 \
    $DECODER_FLAG \
    $FILTER_FLAGS \
    2>&1 | tee logs/maves_${MODEL}_${SLURM_JOB_ID}.log

echo "MAVES training completed!"

echo "============================================"
echo "MAVES Regression Training Completed!"
echo "============================================"