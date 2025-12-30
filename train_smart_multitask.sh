#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_smart_multi
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=450G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:4 # match ddp.yaml num_processes
#SBATCH --ntasks=1 # Keep as 1 since we'll use accelerate launch
#SBATCH --output=logs/smart_multi_output_%j.log
#SBATCH --error=logs/smart_multi_error_%j.log  
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
MODEL="luca"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-star, luca

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

echo "============================================"
echo "Starting SMART Multi-Threshold Training"
echo "Model: $MODEL"
echo "============================================"

# Data source: 'smart' or 'clinvar'
DATA_SOURCE="smart"

# Run multi-task training with all three tasks: MAVES + CLNDN + CLNSIG
echo "=================================="
echo "Starting Multi-Task Training (MAVES + CLNDN + CLNSIG)..."
echo "Data source: $DATA_SOURCE"
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer_smart \
    --model "$MODEL" \
    --seed 127 \
    --multitask \
    --data_source "$DATA_SOURCE" \
    --clndn \
    --clnsig \
    --maves \
    --threshold 65 \
    --learning_rate 0.000005 \
    --batch_size 64 \
    --num_epochs 10 \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_multitask_${SLURM_JOB_ID}.log

echo "Multi-task training completed!"

echo "============================================"
echo "SMART Multi-Threshold Training Completed!"
echo "============================================"
