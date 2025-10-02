#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_continual_model
#SBATCH -p gpu_bwanggroup
#SBATCH --mem=240G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --output=logs/train_output_%j.log
#SBATCH --error=logs/train_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vallisubasri@gmail.com

set -e
mkdir -p logs

cd /cluster/projects/bwanggroup/vsubasri/genomic-FM
source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomic-fm
wandb offline

# Parameters
CHECKPOINT_PATH=${1:-""}
MODEL=${2:-"nt"}
TASK=${3:-"CLNDN"}
BATCH_SIZE=64

# Validate required checkpoint path
if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: CHECKPOINT_PATH must be specified"
    echo "Usage: sbatch train_continual_model.sh <checkpoint_path> [model] [task]"
    echo "Example: sbatch train_continual_model.sh ./root/models/pretrain_model_nt_CLNDN nt CLNDN"
    exit 1
fi

if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: Checkpoint path does not exist: $CHECKPOINT_PATH"
    exit 1
fi

# Set decoder flag for autoregressive models
DECODER_FLAG=""
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
fi

echo "============================================"
echo "Starting SMART Fine-tuning from Checkpoint"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "============================================"

accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29502 \
    heart_finetune_smart.py \
    --model "$MODEL" \
    --task "$TASK" \
    --seed 127 \
    --batch_size $BATCH_SIZE \
    --checkpoint_path "$CHECKPOINT_PATH" \
    $DECODER_FLAG \
    2>&1 | tee logs/smart_${TASK}_from_ckpt_${SLURM_JOB_ID}.log

echo "============================================"
echo "SMART Fine-tuning Completed!"
echo "============================================"
