#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J maves_finetune
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=240G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:4 # match ddp.yaml num_processes
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

################################################################################
# 📋 MAVES FILTERING OPTIONS - Comprehensive Documentation
################################################################################
# Filtering allows you to focus training on specific variant types and conditions
# Multiple filters can be combined for precise dataset control

#=============================================================================
# 1️⃣ GENE FILTERING - Focus on specific genes of interest
#=============================================================================
# Example: Train only on BRCA1, TP53, and EGFR variants
#GENE_FILTER="--filter_genes UBE2I,BRCA1,TP53,EGFR"

#=============================================================================
# 2️⃣ EXPERIMENTAL METHOD FILTERING - Filter by assay type
#=============================================================================
# Available method categories (use category name to include all methods within):
#   DMS                    - Deep mutational scanning (highest quality)
#   REGULATORY             - MPRA, saturation mutagenesis
#   BIOPHYSICAL_STABILITY  - Protease digestion assays (trypsin, chymotrypsin)
#   RNA_ABUNDANCE          - RNA abundance measurements
#   PROTEIN_ABUNDANCE      - Protein abundance assays
#   PROTEIN_TRANSLATION    - Polysome profiling
#   ESCAPE                 - Escape variant assays
#   COMPUTATIONAL_PROCESSED- Enrich2, regression scores (use with caution)
#   EXCLUDE_FROM_TRAINING  - Automatically excluded (control experiments)
#
# Examples:
#METHOD_FILTER="--experimental_methods DMS"                    # All DMS experiments
#METHOD_FILTER="--experimental_methods DMS,REGULATORY"         # DMS + regulatory
#METHOD_FILTER="--experimental_methods DMS-BarSeq,MPRA"        # Specific methods

#=============================================================================
# 3️⃣ VARIANT TYPE FILTERING - Filter by mutation type
#=============================================================================
# Available variant types:
#   sub     - Substitutions/missense variants (SNPs)
#   del     - Deletions
#   ins     - Insertions
#   dup     - Duplications
#   delins  - Deletion-insertions (complex indels)
#   fs      - Frameshift variants
#   inv     - Inversions
#
# Examples:
#VARIANT_FILTER="--variant_types sub"                 # Only substitutions
#VARIANT_FILTER="--variant_types sub,del,ins"         # SNPs and simple indels
#VARIANT_FILTER="--variant_types del,ins,delins"      # All indel types

#=============================================================================
# 4️⃣ REGION TYPE FILTERING - Coding vs non-coding variants
#=============================================================================
# Options:
#   coding     - Only protein-coding variants (HGVS prefix 'c')
#   non-coding - Only non-coding variants (HGVS prefix 'n', 'g')
#   all        - Both coding and non-coding (default)
#
#REGION_FILTER="--region_type coding"      # Protein-coding only
#REGION_FILTER="--region_type non-coding"  # Non-coding only
#REGION_FILTER="--region_type all"         # Both (default)

#=============================================================================
# 5️⃣ SEQUENCE LENGTH FILTERING - Control computational load
#=============================================================================
# Format: --seq_len_min MIN --seq_len_max MAX
# Filters out sequences outside the specified length range
LENGTH_FILTER="--seq_len_min 1 --seq_len_max 1024"

#=============================================================================
# 6️⃣ SAMPLE BALANCING - Prevent dataset imbalance
#=============================================================================
# Limit samples per experiment to avoid bias from large studies
#BALANCE_FILTER="--max_samples_per_experiment 1000"

################################################################################
# 🎯 PRESET FILTER CONFIGURATIONS - Ready-to-use combinations
################################################################################

# === HIGH-QUALITY PROTEIN VARIANTS ===
# Best for protein function prediction models
#FILTER_FLAGS="--experimental_methods DMS --region_type coding --variant_types sub"

# === REGULATORY VARIANTS ===
# For models focusing on gene expression regulation
#FILTER_FLAGS="--experimental_methods REGULATORY --region_type non-coding"

# === STRUCTURAL VARIANTS ===
# Focus on insertions, deletions, and complex rearrangements
#FILTER_FLAGS="--variant_types del,ins,dup,delins,inv"

# === BALANCED MULTI-ASSAY ===
# Diverse training across multiple assay types
#FILTER_FLAGS="--experimental_methods DMS,REGULATORY,RNA_ABUNDANCE --max_samples_per_experiment 500"

# === SPECIFIC GENE STUDY ===
# Deep dive into specific genes of interest
#FILTER_FLAGS="--filter_genes BRCA1,BRCA2 --experimental_methods DMS --variant_types sub,del,ins"

# === CODING SNPs ONLY ===
# Classic missense variant prediction
#FILTER_FLAGS="--region_type coding --variant_types sub --experimental_methods DMS"

# === NON-CODING REGULATORY ===
# Promoter and enhancer variants
#FILTER_FLAGS="--region_type non-coding --experimental_methods REGULATORY,RNA_ABUNDANCE"

################################################################################
# 🚀 ACTIVE CONFIGURATION - Modify this for your training run
################################################################################
FILTER_FLAGS="--experimental_methods DMS"  # Default: High-quality DMS experiments

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
    $LENGTH_FILTER \
    2>&1 | tee logs/maves_${MODEL}_${SLURM_JOB_ID}.log

echo "MAVES training completed!"

echo "============================================"
echo "MAVES Regression Training Completed!"
echo "============================================"
