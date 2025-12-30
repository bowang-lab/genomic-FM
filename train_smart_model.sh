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

# MAVES filtering options
MAVES_FILTER_FLAGS="--experimental_methods DMS"
MAVES_LENGTH_FILTER="--seq_len_min 1 --seq_len_max 1024"

# Run MAVES regression training first
echo "=================================="
echo "Starting MAVES score regression training (DMS)..."
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29498 \
    -m src.pack_tunable_model.hf_trainer \
    --model "$MODEL" \
    --task MAVES \
    --seed 127 \
    --learning_rate 0.000005 \
    --batch_size 64 \
    --num_epochs 10 \
    --max_grad_norm 1.0 \
    --num_workers 8 \
    $DECODER_FLAG \
    $MAVES_FILTER_FLAGS \
    $MAVES_LENGTH_FILTER \
    2>&1 | tee logs/smart_MAVES_DMS_${SLURM_JOB_ID}.log

echo "MAVES DMS training completed!"

# Run multi-threshold training for CLNDN (disease classification)
echo "=================================="
echo "Starting multi-threshold CLNDN task training (disease classification)..."
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29500 heart_finetune_smart.py \
    --model "$MODEL" \
    --seed 127 \
    --task CLNDN \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_multi_CLNDN_${SLURM_JOB_ID}.log

echo "Multi-threshold CLNDN training completed!"

# Run multi-threshold training for CLNSIG (pathogenicity classification)
echo "=================================="
echo "Starting multi-threshold CLNSIG task training (pathogenicity classification)..."
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29501 heart_finetune_smart.py \
    --model "$MODEL" \
    --seed 127 \
    --task CLNSIG \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_multi_CLNSIG_${SLURM_JOB_ID}.log

echo "Multi-threshold CLNSIG training completed!"

echo "============================================"
echo "SMART Multi-Threshold Training Completed!"
echo "============================================"
