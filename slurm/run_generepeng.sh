#!/bin/bash
#SBATCH -t 2-00:0:0
#SBATCH -J gene_repeng
#SBATCH -p gpu_bwanggroup
#SBATCH --account=bwanggroup_gpu
#SBATCH --mem=200G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=logs/repeng_output_%j.log
#SBATCH --error=logs/repeng_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vallisubasri@gmail.com

# Exit on any error
set -e

cd /cluster/projects/bwanggroup/vsubasri/genomic-FM

# Create output directories if they don't exist
mkdir -p logs root/output

source ~/miniconda3/etc/profile.d/conda.sh

conda activate genomic-fm

# Disable wandb syncing for now
wandb offline

# ============================================
# CONFIGURATION
# ============================================
# Model path: can be a model name in root/models/ or full path
# Examples:
#   - "omni_dna_116m" (default base model)
#   - "pretrain_model_omni_dna_116m_CLNDN" (finetuned model)
#   - "pretrain_model_dnabert2_CLNDN"
#   - "root/models/pretrain_model_nt_CLNDN" (full path)
MODEL_PATH="${1:-omni_dna_116m}"  # Default to omni_dna_116m, or use first argument
DATASET="cgc_primary_findings"  # CGC pediatric cardiac patient variants

echo "============================================"
echo "Gene Representation Engineering"
echo "Model Path: $MODEL_PATH"
echo "Dataset: $DATASET (CGC Cardiac)"
echo "============================================"

# Run example usage script with model path
python -m src.geneRepEng.example_usage \
    --model_path "$MODEL_PATH" \
    2>&1 | tee logs/repeng_${MODEL_PATH//\//_}_${SLURM_JOB_ID}.log

echo "============================================"
echo "Gene Representation Engineering Completed!"
echo "Control vector saved: root/output/cgc_cardiac_pathogenicity_control_${MODEL_PATH//\//_}.npz"
echo "============================================"
