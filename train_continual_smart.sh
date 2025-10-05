#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_smart_fromckpt
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=400G # at most 450G
#SBATCH -c 8 # at most 60
#SBATCH -N 1 # number of node
#SBATCH --gres=gpu:4 # match ddp.yaml num_processes
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
MODEL="nt"  # Options: nt, omni_dna_116m, hyenadna, caduceus, gena-lm, dnabert2, gpn-star

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

# Checkpoint paths
CLNDN_CHECKPOINT="root/models/pretrain_model_${MODEL}_CLNDN"
CLNSIG_CHECKPOINT="root/models/pretrain_model_${MODEL}_CLNSIG"

# Tasks to train on
TASKS="CLNDN CLNSIG"

# Thresholds to train
THRESHOLDS="50.0 55.0 60.0 65.0 70.0"

echo "============================================"
echo "Starting SMART Multi-Threshold Training from Checkpoints"
echo "Model: $MODEL"
echo "CLNDN Checkpoint: $CLNDN_CHECKPOINT"
echo "CLNSIG Checkpoint: $CLNSIG_CHECKPOINT"
echo "Tasks: $TASKS"
echo "Thresholds: $THRESHOLDS"
echo "============================================"

# Counter for port assignment
PORT_COUNTER=0

# Loop through each checkpoint
for CHECKPOINT_NAME in "CLNDN" "CLNSIG"; do
    if [ "$CHECKPOINT_NAME" == "CLNDN" ]; then
        CHECKPOINT="$CLNDN_CHECKPOINT"
    else
        CHECKPOINT="$CLNSIG_CHECKPOINT"
    fi

    echo ""
    echo "========================================================================"
    echo "Training from checkpoint: $CHECKPOINT_NAME ($CHECKPOINT)"
    echo "========================================================================"

    # Loop through each task
    for TASK in $TASKS; do
        echo ""
        echo "===================================="
        echo "Task: $TASK"
        echo "===================================="

        # Loop through each threshold
        for THRESHOLD in $THRESHOLDS; do
            PORT=$((29700 + PORT_COUNTER))
            PORT_COUNTER=$((PORT_COUNTER + 1))

            echo ""
            echo "============================================================"
            echo "Checkpoint: $CHECKPOINT_NAME | Task: $TASK | Threshold: $THRESHOLD | Port: $PORT"
            echo "============================================================"

            accelerate launch \
                --config_file configs/ddp.yaml \
                --main_process_port $PORT \
                -m src.pack_tunable_model.hf_trainer_smart \
                --threshold "$THRESHOLD" \
                --model "$MODEL" \
                --seed 127 \
                --task "$TASK" \
                --checkpoint_path "$CHECKPOINT" \
                $DECODER_FLAG \
                2>&1 | tee logs/smart_from_${CHECKPOINT_NAME}_to_${TASK}_threshold_${THRESHOLD}_${SLURM_JOB_ID}.log

            if [ $? -eq 0 ]; then
                echo "✓ Training completed: $CHECKPOINT_NAME → $TASK (threshold $THRESHOLD)"
            else
                echo "✗ Training failed: $CHECKPOINT_NAME → $TASK (threshold $THRESHOLD)"
                exit 1
            fi
        done
    done
done

echo ""
echo "============================================"
echo "All SMART Multi-Threshold Training from Checkpoints Completed!"
echo "============================================"