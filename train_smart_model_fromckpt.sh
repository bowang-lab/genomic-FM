#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_smart_fromckpt
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=240G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:2 # match ddp.yaml num_processes
#SBATCH --ntasks=1 # Keep as 1 since we'll use accelerate launch
#SBATCH --output=logs/smart_fromckpt_output_%j.log
#SBATCH --error=logs/smart_fromckpt_error_%j.log  
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
MODEL="nt"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-msa-sapiens

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

# Checkpoint paths
CLNDN_CHECKPOINT="root/output/pretrain_model_${MODEL}_CLNDN"
CLNSIG_CHECKPOINT="root/output/pretrain_model_${MODEL}_CLNSIG"

echo "============================================"
echo "Starting SMART Multi-Threshold Training from Checkpoints"
echo "Model: $MODEL"
echo "CLNDN Checkpoint: $CLNDN_CHECKPOINT"
echo "CLNSIG Checkpoint: $CLNSIG_CHECKPOINT"
echo "============================================"

# Run multi-threshold training for CLNDN (disease classification) with checkpoint
echo "=================================="
echo "Starting multi-threshold CLNDN task training (disease classification) from checkpoint..."
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29600 heart_finetune_multi_smart.py \
    --model "$MODEL" \
    --seed 127 \
    --task CLNDN \
    --checkpoint_path "$CLNDN_CHECKPOINT" \
    --continue_on_error \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_multi_CLNDN_fromckpt_${SLURM_JOB_ID}.log

echo "Multi-threshold CLNDN training from checkpoint completed!"

# Run multi-threshold training for CLNSIG (pathogenicity) with checkpoint
echo "=================================="
echo "Starting multi-threshold CLNSIG task training (pathogenicity classification) from checkpoint..."
echo "=================================="
accelerate launch --config_file configs/ddp.yaml --main_process_port 29601 heart_finetune_multi_smart.py \
    --model "$MODEL" \
    --seed 127 \
    --task CLNSIG \
    --checkpoint_path "$CLNSIG_CHECKPOINT" \
    --continue_on_error \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_multi_CLNSIG_fromckpt_${SLURM_JOB_ID}.log

echo "Multi-threshold CLNSIG training from checkpoint completed!"

echo "============================================"
echo "SMART Multi-Threshold Training from Checkpoints Completed!"
echo "============================================"