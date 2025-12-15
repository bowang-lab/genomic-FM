#!/bin/bash
#SBATCH -t 2-00:0:0
#SBATCH -J gene_repeng
#SBATCH -p gpu_bwanggroup
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

# Create logs directory if it doesn't exist
mkdir -p logs

cd /cluster/projects/bwanggroup/vsubasri/genomic-FM

source ~/miniconda3/etc/profile.d/conda.sh

conda activate genomic-fm

# Disable wandb syncing for now
wandb offline

# ============================================
# CONFIGURATION
# ============================================
# Model options: omni_dna_116m, dnabert2, nt, gena-lm, gpn-msa-sapiens
MODEL="omni_dna_116m"
DATASET="cgc_primary_findings"  # CGC pediatric cardiac patient variants

echo "============================================"
echo "Gene Representation Engineering"
echo "Model: $MODEL"
echo "Dataset: $DATASET (CGC Cardiac)"
echo "============================================"

# Run example usage script
python -m src.geneRepEng.example_usage \
    2>&1 | tee logs/repeng_${MODEL}_${SLURM_JOB_ID}.log

echo "============================================"
echo "Gene Representation Engineering Completed!"
echo "Control vector saved: cgc_cardiac_pathogenicity_control.npz"
echo "============================================"
