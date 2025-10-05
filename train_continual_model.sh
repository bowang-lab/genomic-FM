#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_continual
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=400G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:4 # match ddp.yaml num_processes
#SBATCH --ntasks=1 # Keep as 1 since we'll use accelerate launch
#SBATCH --output=logs/continual_output_%j.log
#SBATCH --error=logs/continual_error_%j.log
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

# ============================================
# TRAINING PARAMETERS - MODIFY THESE
# ============================================

# Model architecture to use
MODEL="nt"  # Options: nt, hyenadna, omni_dna_116m, etc.

# Checkpoint to load from (relative to root/models/ or absolute path)
CHECKPOINT="pretrain_model_nt_MAVES_score_DMS"

# Target task to train on
TASK="CLNDN"  # Options: CLNDN, CLNSIG, MAVES

# Batch size
BATCH_SIZE=64

# ============================================

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

echo "============================================"
echo "Starting Continual Training from Checkpoint"
echo "Model Type: $MODEL"
echo "Checkpoint: $CHECKPOINT"
echo "Target Task: $TASK"
echo "Batch Size: $BATCH_SIZE"
echo "============================================"

accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29502 \
    -m src.pack_tunable_model.hf_trainer \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    --batch_size $BATCH_SIZE \
    --pretrained_model "$CHECKPOINT" \
    $DECODER_FLAG \
    2>&1 | tee logs/${TASK}_from_$(basename ${CHECKPOINT})_${SLURM_JOB_ID}.log

if [ $? -eq 0 ]; then
    echo "✓ Continual training completed successfully"
else
    echo "✗ Continual training failed"
    exit 1
fi

echo "============================================"
echo "Continual Training Completed!"
echo "============================================"
