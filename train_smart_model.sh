#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_smart_multi
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=240G # at most 450G
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
MODEL="nt"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-star

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

# Run multi-threshold training for CLNDN (disease classification)
echo "=================================="
echo "Starting multi-threshold CLNDN task training (disease classification)..."
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29500 heart_finetune_multi_smart.py \
    --model "$MODEL" \
    --seed 127 \
    --task CLNDN \
    --continue_on_error \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_multi_CLNDN_${SLURM_JOB_ID}.log

echo "Multi-threshold CLNDN training completed!"

# Run multi-threshold training for CLNSIG (pathogenicity classification)
echo "=================================="
echo "Starting multi-threshold CLNSIG task training (pathogenicity classification)..."
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29501 heart_finetune_multi_smart.py \
    --model "$MODEL" \
    --seed 127 \
    --task CLNSIG \
    --continue_on_error \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_multi_CLNSIG_${SLURM_JOB_ID}.log

echo "Multi-threshold CLNSIG training completed!"

echo "============================================"
echo "SMART Multi-Threshold Training Completed!"
echo "============================================"
