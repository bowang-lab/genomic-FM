#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_smart_disc
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=450G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --output=logs/smart_disc_output_%j.log
#SBATCH --error=logs/smart_disc_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vallisubasri@gmail.com

# =============================================================================
# Discriminative Multitask Training (Task-Routing with Classification Heads)
# =============================================================================
#
# Each sample is routed to a task-specific classification head.
# Uses CrossEntropy loss for classification tasks.
#
# =============================================================================

set -e

# Create logs directory if it doesn't exist
mkdir -p logs

cd /cluster/projects/bwanggroup/vsubasri/genomic-FM

source ~/miniconda3/etc/profile.d/conda.sh

conda activate genomic-fm

# Disable wandb syncing for now
wandb offline

# Training parameters
MODEL="omni_dna_116m"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-star, luca

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

echo "============================================"
echo "Discriminative Multitask Training"
echo "============================================"
echo "Model: $MODEL"
echo "Tasks: CLNDN + CLNSIG"
echo "============================================"

# Data source: 'smart' or 'clinvar'
DATA_SOURCE="smart"

accelerate launch --config_file configs/ddp.yaml --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer_smart \
    --model "$MODEL" \
    --seed 127 \
    --multitask \
    --data_source "$DATA_SOURCE" \
    --clndn \
    --clnsig \
    --threshold 65 \
    --learning_rate 0.000005 \
    --batch_size 8 \
    --num_epochs 10 \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_discriminative_${SLURM_JOB_ID}.log

echo ""
echo "============================================"
echo "Discriminative Training Completed!"
echo "============================================"
