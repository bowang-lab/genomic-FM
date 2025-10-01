#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_continual_model
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=240G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:4 # match ddp.yaml num_processes
#SBATCH --ntasks=1 # Keep as 1 since we'll use accelerate launch
#SBATCH --output=logs/train_output_%j.log 
#SBATCH --error=logs/train_error_%j.log  
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vallisubasri@gmail.com

# Exit on any error
set -e

# =============================================================================
# CONTINUAL LEARNING SCRIPT
# =============================================================================
# This script performs continual learning from existing trained models to SMART variants.
# It resumes training from ClinVar, MAVES, or other previously trained checkpoints.
#
# Usage:
#   sbatch train_continual_model.sh <base_task> <base_subtask> [checkpoint_path]
#
# Parameters:
#   $1: BASE_TASK (e.g., "clinvar", "maves" - required)
#   $2: BASE_SUBTASK (e.g., "CLNDN", "CLNSIG" - required)
#   $3: CHECKPOINT_PATH (optional, auto-detects if not provided)
#
# Examples:
#   # Continue from ClinVar CLNDN model
#   sbatch train_continual_model.sh clinvar CLNDN
#
#   # Continue from MAVES CLNSIG model with specific checkpoint
#   sbatch train_continual_model.sh maves CLNSIG ./results/omni_dna_116m_maves_CLNSIG/checkpoint-1000
#
#   # Continue from ClinVar CLNSIG (auto-detect checkpoint)
#   sbatch train_continual_model.sh clinvar CLNSIG
# =============================================================================

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
MODEL="omni_dna_116m"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-star
WANDB_PROJECT="genomic-finetune-clinvar"
BATCH_SIZE=64  # Batch size per GPU

# Continual learning parameters (required)
BASE_TASK=${1:-""}  # Base task to continue from (e.g., "clinvar", "maves")
BASE_SUBTASK=${2:-""}  # Base subtask (e.g., "CLNDN", "CLNSIG")
CHECKPOINT_PATH=${3:-""}  # Path to existing checkpoint (optional, will auto-detect if not provided)

# Validate required parameters
if [[ -z "$BASE_TASK" ]]; then
    echo "ERROR: BASE_TASK must be specified (e.g., 'clinvar', 'maves')"
    echo "Usage: sbatch train_continual_model.sh <base_task> <base_subtask> [checkpoint_path]"
    exit 1
fi
if [[ -z "$BASE_SUBTASK" ]]; then
    echo "ERROR: BASE_SUBTASK must be specified (e.g., 'CLNDN', 'CLNSIG')"
    echo "Usage: sbatch train_continual_model.sh <base_task> <base_subtask> [checkpoint_path]"
    exit 1
fi

echo "=== Continual Learning Configuration ==="
echo "Base task: $BASE_TASK"
echo "Base subtask: $BASE_SUBTASK"
if [[ -n "$CHECKPOINT_PATH" ]]; then
    echo "Checkpoint path: $CHECKPOINT_PATH"
else
    echo "Checkpoint path: Auto-detect from base task"
fi
echo "========================================"

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

# Function to auto-detect checkpoint path
auto_detect_checkpoint() {
    local base_task=$1
    local base_subtask=$2
    local model=$3

    # Common checkpoint locations based on typical training patterns
    local possible_paths=(
        "results/${model}_${base_task}_${base_subtask}/checkpoint-*"
        "checkpoints/${model}_${base_task}_${base_subtask}/checkpoint-*"
        "output/${model}_${base_task}_${base_subtask}/checkpoint-*"
        "models/${model}_${base_task}_${base_subtask}/checkpoint-*"
    )

    for pattern in "${possible_paths[@]}"; do
        # Find the most recent checkpoint
        local latest_checkpoint=$(find . -path "./$pattern" -type d 2>/dev/null | sort -V | tail -1)
        if [[ -n "$latest_checkpoint" ]]; then
            echo "$latest_checkpoint"
            return 0
        fi
    done

    echo ""
    return 1
}

# Set checkpoint path for continual learning
if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "Auto-detecting checkpoint path for ${BASE_TASK}_${BASE_SUBTASK}..."
    CHECKPOINT_PATH=$(auto_detect_checkpoint "$BASE_TASK" "$BASE_SUBTASK" "$MODEL")
    if [[ -z "$CHECKPOINT_PATH" ]]; then
        echo "ERROR: Could not auto-detect checkpoint path for ${BASE_TASK}_${BASE_SUBTASK}"
        echo "Please provide explicit CHECKPOINT_PATH as 3rd argument"
        exit 1
    else
        echo "Found checkpoint: $CHECKPOINT_PATH"
    fi
fi

# Verify checkpoint exists
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

CHECKPOINT_FLAG="--resume_from_checkpoint $CHECKPOINT_PATH"

# ============================================
# CONTINUAL LEARNING TRAINING LOGIC
# ============================================

if [[ "$BASE_TASK" == "maves" ]]; then
    # ============================================
    # MAVES TO SMART CONTINUAL LEARNING
    # ============================================
    echo "============================================"
    echo "Starting MAVES → SMART Continual Learning"
    echo "Base model: ${BASE_TASK}_${BASE_SUBTASK}"
    echo "============================================"
else
    # ============================================
    # CLINVAR TO SMART CONTINUAL LEARNING
    # ============================================
    echo "============================================"
    echo "Starting ClinVar → SMART Continual Learning"
    echo "Base model: ${BASE_TASK}_${BASE_SUBTASK}"
    echo "============================================"
fi

# Run training for CLNDN task (disease classification) on SMART variants
echo "=================================="
echo "Starting SMART CLNDN continual learning from ${BASE_TASK}_${BASE_SUBTASK}..."
echo "=================================="
TASK="CLNDN"
accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29502 \
    heart_finetune_smart.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    --batch_size $BATCH_SIZE \
    $DECODER_FLAG \
    $CHECKPOINT_FLAG \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_smart_CLNDN_from_${BASE_TASK}_${BASE_SUBTASK}.log

echo "SMART CLNDN training completed!"

# Run training for CLNSIG task (pathogenicity classification) on SMART variants
echo "=================================="
echo "Starting SMART CLNSIG continual learning from ${BASE_TASK}_${BASE_SUBTASK}..."
echo "=================================="
TASK="CLNSIG"

# Update checkpoint if we're switching tasks
if [[ "$BASE_SUBTASK" != "CLNSIG" ]]; then
    echo "Updating checkpoint path for CLNSIG task..."
    CLNSIG_CHECKPOINT_PATH=$(auto_detect_checkpoint "$BASE_TASK" "CLNSIG" "$MODEL")
    if [[ -n "$CLNSIG_CHECKPOINT_PATH" ]]; then
        CHECKPOINT_FLAG="--resume_from_checkpoint $CLNSIG_CHECKPOINT_PATH"
        echo "Using CLNSIG checkpoint: $CLNSIG_CHECKPOINT_PATH"
    else
        echo "No CLNSIG checkpoint found, using original: $CHECKPOINT_PATH"
    fi
fi

accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29503 \
    heart_finetune_smart.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    --batch_size $BATCH_SIZE \
    $DECODER_FLAG \
    $CHECKPOINT_FLAG \
    2>&1 | tee logs/training_${SLURM_JOB_ID}_smart_CLNSIG_from_${BASE_TASK}_${BASE_SUBTASK}.log

echo "SMART CLNSIG training completed!"

echo "============================================"
echo "All continual learning tasks completed successfully!"
echo "============================================"
echo ""
echo "Training Summary:"
echo "Mode: Continual Learning from ${BASE_TASK}_${BASE_SUBTASK}"
echo "1. SMART variant CLNDN training: Complete"
echo "2. SMART variant CLNSIG training: Complete"
echo ""
echo "Check the following log files for details:"
echo "- logs/training_${SLURM_JOB_ID}_smart_CLNDN_from_${BASE_TASK}_${BASE_SUBTASK}.log"
echo "- logs/training_${SLURM_JOB_ID}_smart_CLNSIG_from_${BASE_TASK}_${BASE_SUBTASK}.log"