#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_clnvar_model
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=240G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:4 # at most 4
#SBATCH --ntasks=1 # Keep as 1 since we'll use accelerate launch
#SBATCH --output=logs/train_output_%j.log 
#SBATCH --error=logs/train_error_%j.log  
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

# Print distributed setup for debugging
echo "=== Distributed Training Setup ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=================================="

# Training parameters
MODEL="nt"  # Options: nt, dnabert2, olmo
WANDB_PROJECT="genomic-finetune-clinvar"

# Run training for CLNDN task (pathogenic vs benign)
echo "=================================="
echo "Starting CLNDN task training..."
echo "=================================="
TASK="CLNDN"
accelerate launch \
    --config_file configs/ddp.yaml \
    heart_finetune.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_CLNDN.log

echo "CLNDN training completed!"

# Run training for CLNSIG task (clinical significance)
echo "=================================="
echo "Starting CLNSIG task training..."
echo "=================================="
TASK="CLNSIG"
accelerate launch \
    --config_file configs/ddp.yaml \
    heart_finetune.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_CLNSIG.log

echo "CLNSIG training completed!"
echo "All training tasks completed successfully!"