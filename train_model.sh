#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_clnvar_model
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=240G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:2 # match ddp.yaml num_processes
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
MODEL="omni_dna_116m"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm (dnabert2 has Triton issues)
WANDB_PROJECT="genomic-finetune-clinvar"

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

# ============================================
# PART 1: GENERAL CLINVAR TRAINING
# ============================================
echo "============================================"
echo "PART 1: Starting General ClinVar Training"
echo "============================================"

# Run training for CLNDN task (pathogenic vs benign) on general ClinVar
echo "=================================="
echo "Starting general ClinVar CLNDN task training..."
echo "=================================="
TASK="CLNDN"
accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29500 \
    heart_finetune.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    $DECODER_FLAG \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_general_CLNDN.log

echo "General ClinVar CLNDN training completed!"

# Run training for CLNSIG task (clinical significance) on general ClinVar
echo "=================================="
echo "Starting general ClinVar CLNSIG task training..."
echo "=================================="
TASK="CLNSIG"
accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29501 \
    heart_finetune.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    $DECODER_FLAG \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_general_CLNSIG.log

echo "General ClinVar CLNSIG training completed!"

# ============================================
# PART 2: SMART VARIANT HEART-SPECIFIC TRAINING
# ============================================
echo "============================================"
echo "PART 2: Starting SMART Variant Training (Heart-specific)"
echo "============================================"

# Run training for CLNDN task (pathogenic vs benign) on SMART variants
echo "=================================="
echo "Starting SMART CLNDN task training..."
echo "=================================="
TASK="CLNDN"
accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29502 \
    heart_finetune_smart.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    $DECODER_FLAG \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_smart_CLNDN.log

echo "SMART CLNDN training completed!"

# Run training for CLNSIG task (clinical significance) on SMART variants
echo "=================================="
echo "Starting SMART CLNSIG task training..."
echo "=================================="
TASK="CLNSIG"
accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29503 \
    heart_finetune_smart.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    $DECODER_FLAG \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_smart_CLNSIG.log

echo "SMART CLNSIG training completed!"

echo "============================================"
echo "All training tasks completed successfully!"
echo "============================================"
echo ""
echo "Training Summary:"
echo "1. General ClinVar training: Complete"
echo "2. SMART variant CLNDN training: Complete"
echo "3. SMART variant CLNSIG training: Complete"
echo ""
echo "Check the following log files for details:"
echo "- logs/training_${SLURM_JOB_ID}_general_CLNSIG.log"
echo "- logs/training_${SLURM_JOB_ID}_smart_CLNDN.log"
echo "- logs/training_${SLURM_JOB_ID}_smart_CLNSIG.log"