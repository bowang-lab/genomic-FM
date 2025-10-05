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
# TRAINING PARAMETERS
# ============================================

# Model architecture to use
MODEL="nt"  # Options: nt, hyenadna, omni_dna_116m, etc.

# Checkpoint to load from (relative to root/models/ or absolute path)
CHECKPOINT="pretrain_model_nt_MAVES_score_DMS"

# Target tasks to train on (space-separated list)
TASKS="CLNDN CLNSIG"  # Options: CLNDN, CLNSIG, MAVES (or any combination)

# Training mode
TRAINING_MODE="regular"  # Options: regular, smart

# Thresholds for SMART mode (only used when TRAINING_MODE=smart)
THRESHOLDS="50.0 55.0 60.0 65.0 70.0"

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
echo "Target Tasks: $TASKS"
echo "Training Mode: $TRAINING_MODE"
if [ "$TRAINING_MODE" == "smart" ]; then
    echo "Thresholds: $THRESHOLDS"
fi
echo "Batch Size: $BATCH_SIZE"
echo "============================================"

# Convert tasks string to array
TASK_ARRAY=($TASKS)

# Base port number
BASE_PORT=29502

# Port counter for unique ports
PORT_COUNTER=0

if [ "$TRAINING_MODE" == "regular" ]; then
    # Regular mode: train each task once
    for TASK in "${TASK_ARRAY[@]}"; do
        PORT=$((BASE_PORT + PORT_COUNTER))
        PORT_COUNTER=$((PORT_COUNTER + 1))

        echo ""
        echo "============================================================"
        echo "Training on task: $TASK (Port: $PORT)"
        echo "============================================================"

        accelerate launch \
            --config_file configs/ddp.yaml \
            --main_process_port $PORT \
            -m src.pack_tunable_model.hf_trainer \
            --model "$MODEL" \
            --task "$TASK" \
            --seed 127 \
            --batch_size $BATCH_SIZE \
            --pretrained_model "$CHECKPOINT" \
            $DECODER_FLAG \
            2>&1 | tee logs/${TASK}_from_$(basename ${CHECKPOINT})_${SLURM_JOB_ID}.log

        if [ $? -eq 0 ]; then
            echo "✓ Training on task $TASK completed successfully"
        else
            echo "✗ Training on task $TASK failed"
            exit 1
        fi
    done

elif [ "$TRAINING_MODE" == "smart" ]; then
    # SMART mode: train each task with each threshold
    for TASK in "${TASK_ARRAY[@]}"; do
        echo ""
        echo "===================================="
        echo "Task: $TASK"
        echo "===================================="

        for THRESHOLD in $THRESHOLDS; do
            PORT=$((BASE_PORT + PORT_COUNTER))
            PORT_COUNTER=$((PORT_COUNTER + 1))

            echo ""
            echo "============================================================"
            echo "Task: $TASK | Threshold: $THRESHOLD | Port: $PORT"
            echo "============================================================"

            accelerate launch \
                --config_file configs/ddp.yaml \
                --main_process_port $PORT \
                -m src.pack_tunable_model.hf_trainer_smart \
                --model "$MODEL" \
                --task "$TASK" \
                --seed 127 \
                --batch_size $BATCH_SIZE \
                --threshold "$THRESHOLD" \
                --checkpoint_path "$CHECKPOINT" \
                $DECODER_FLAG \
                2>&1 | tee logs/smart_${TASK}_threshold_${THRESHOLD}_from_$(basename ${CHECKPOINT})_${SLURM_JOB_ID}.log

            if [ $? -eq 0 ]; then
                echo "✓ Training completed: $TASK (threshold $THRESHOLD)"
            else
                echo "✗ Training failed: $TASK (threshold $THRESHOLD)"
                exit 1
            fi
        done
    done
else
    echo "Error: Invalid TRAINING_MODE '$TRAINING_MODE'. Must be 'regular' or 'smart'."
    exit 1
fi

echo ""
echo "============================================"
echo "Continual Training Completed!"
echo "============================================"
