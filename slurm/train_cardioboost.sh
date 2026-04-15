#!/bin/bash
#SBATCH -t 1-00:0:0
#SBATCH -J train_cardioboost
#SBATCH -p gpu_bwanggroup
#SBATCH --account=bwanggroup_gpu
#SBATCH --mem=80G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output=logs/cardioboost_output_%j.log
#SBATCH --error=logs/cardioboost_error_%j.log
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

MODEL="omni_dna_116m"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2

# Disease type: cm (cardiomyopathy) or arm (arrhythmia)
DISEASE_TYPE="${1:-cm}"

# Sequence length - default 1024 to match CGC/SMART
SEQ_LENGTH="${2:-1024}"

# Whether to include VUS data in training (true/false)
INCLUDE_VUS="${3:-false}"

# Data directory (relative to genomic-FM)
DATA_DIR="root/data/CardioBoost"

# Training hyperparameters
LEARNING_RATE=0.00001
BATCH_SIZE=16
NUM_EPOCHS=50

# ============================================

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

# Set VUS flag
if [[ "$INCLUDE_VUS" == "true" ]]; then
    VUS_FLAG="--include_vus"
else
    VUS_FLAG=""
fi

echo "============================================"
echo "Starting CardioBoost Training"
echo "Model: $MODEL"
echo "Disease Type: $DISEASE_TYPE"
echo "Sequence Length: $SEQ_LENGTH"
echo "Include VUS: $INCLUDE_VUS"
echo "Data Directory: $DATA_DIR"
echo "============================================"

python -m src.pack_tunable_model.hf_trainer_cardioboost \
    --model "$MODEL" \
    --disease_type "$DISEASE_TYPE" \
    --data_dir "$DATA_DIR" \
    --seq_length "$SEQ_LENGTH" \
    --seed 127 \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    $DECODER_FLAG \
    $VUS_FLAG \
    2>&1 | tee logs/cardioboost_${DISEASE_TYPE}_${SEQ_LENGTH}_${SLURM_JOB_ID}.log

echo "============================================"
echo "CardioBoost Training Completed!"
echo "Disease Type: $DISEASE_TYPE"
echo "Sequence Length: $SEQ_LENGTH"
echo "============================================"
