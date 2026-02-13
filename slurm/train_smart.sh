#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_smart
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=450G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --output=logs/smart_output_%j.log
#SBATCH --error=logs/smart_error_%j.log
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
# TRAINING PARAMETERS
# ============================================

MODEL="omni_dna_116m"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-star, luca

# Task to train on
TASK="CLNDN"  # Options: CLNDN (disease), CLNSIG (pathogenicity)

# Threshold for SMART score filtering
THRESHOLD=65

# Training hyperparameters
LEARNING_RATE=0.000005
BATCH_SIZE=8
NUM_EPOCHS=10

# ============================================

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

echo "============================================"
echo "Starting SMART Single-Task Training"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Threshold: $THRESHOLD"
echo "============================================"

accelerate launch --config_file configs/ddp.yaml --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer_smart \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    --threshold "$THRESHOLD" \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_${TASK}_threshold_${THRESHOLD}_${SLURM_JOB_ID}.log

echo "============================================"
echo "SMART Training Completed!"
echo "Task: $TASK"
echo "Threshold: $THRESHOLD"
echo "============================================"
