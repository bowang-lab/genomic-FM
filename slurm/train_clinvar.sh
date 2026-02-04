#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_clinvar
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=450G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of nodes
#SBATCH --gres=gpu:4 # match ddp.yaml num_processes
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
MODEL="luca"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-star, luca
WANDB_PROJECT="genomic-finetune-clinvar"
BATCH_SIZE=8  # Batch size per GPU

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

# ============================================
# CLINVAR TRAINING
# ============================================
echo "============================================"
echo "Starting ClinVar Training"
echo "============================================"

# Run training for CLNDN task (disease classification) on ClinVar
echo "=================================="
echo "Starting ClinVar CLNDN task training..."
echo "=================================="
TASK="CLNDN"
# To filter by disease subset, add: --disease_subset_file heart_related_diseases.txt
accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    $DECODER_FLAG \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_clinvar_CLNDN.log

echo "ClinVar CLNDN training completed!"

# Run training for CLNSIG task (pathogenicity classification) on ClinVar
echo "=================================="
echo "Starting ClinVar CLNSIG task training..."
echo "=================================="
TASK="CLNSIG"
accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29501 \
    -m src.pack_tunable_model.hf_trainer \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    $DECODER_FLAG \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_clinvar_CLNSIG.log

echo "ClinVar CLNSIG training completed!"

echo "============================================"
echo "All ClinVar training tasks completed successfully!"
echo "============================================"
echo ""
echo "Training Summary:"
echo "1. ClinVar CLNDN training: Complete"
echo "2. ClinVar CLNSIG training: Complete"
echo ""
echo "Check the following log files for details:"
echo "- logs/training_${SLURM_JOB_ID}_clinvar_CLNDN.log"
echo "- logs/training_${SLURM_JOB_ID}_clinvar_CLNSIG.log"