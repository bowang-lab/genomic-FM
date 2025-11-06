#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_clinvar
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=450G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 2 # number of nodes
#SBATCH --gres=gpu:4 # 4 GPUs per node = 8 total GPUs
#SBATCH --ntasks=2 # 1 task per node for multi-node training
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

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Multi-node setup for accelerate
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Resolve to IPv4 address to avoid IPv6 issues
export MASTER_ADDR=$(getent ahosts $MASTER_NODE | grep STREAM | head -n 1 | awk '{print $1}')
export MASTER_PORT=29500

# Force IPv4 and configure network for distributed training
export NCCL_SOCKET_IFNAME=^docker0,lo
export GLOO_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# Force IPv4 for PyTorch distributed (fixes IPv6 resolution errors)
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_FAMILY=AF_INET

# Determine machine rank (0 for first node, 1 for second, etc.)
export MACHINE_RANK=${SLURM_NODEID:-0}

# Print distributed setup for debugging
echo "=== Distributed Training Setup ==="
echo "Hostname: $(hostname)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MASTER_NODE: $MASTER_NODE"
echo "MASTER_ADDR (IPv4): $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "MACHINE_RANK: $MACHINE_RANK"
echo "=================================="

# Create a temporary accelerate config with the correct IP and rank
TEMP_CONFIG="configs/ddp_${SLURM_JOB_ID}_${MACHINE_RANK}.yaml"
sed -e "s/MASTER_ADDR_PLACEHOLDER/$MASTER_ADDR/" \
    -e "s/MACHINE_RANK_PLACEHOLDER/$MACHINE_RANK/" \
    configs/ddp.yaml > $TEMP_CONFIG

echo "Generated config for rank $MACHINE_RANK:"
cat $TEMP_CONFIG
echo "=================================="

# Training parameters
MODEL="luca"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-star, luca
WANDB_PROJECT="genomic-finetune-clinvar"
BATCH_SIZE=4  # Batch size per GPU - reduced for large models like LucaOne

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
    --config_file $TEMP_CONFIG \
    heart_finetune.py \
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
    --config_file $TEMP_CONFIG \
    heart_finetune.py \
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