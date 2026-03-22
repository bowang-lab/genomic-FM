#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_smart_gen
#SBATCH -p gpu_bwanggroup
#SBATCH --account=bwanggroup_gpu
#SBATCH --mem=450G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --output=logs/smart_gen_output_%j.log
#SBATCH --error=logs/smart_gen_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vallisubasri@gmail.com

# =============================================================================
# Generative Multitask Training (SFTTrainer)
# =============================================================================
#
# Uses SFTTrainer from trl for generative fine-tuning.
# Only works with decoder-only models (OmniDNA).
#
# Format: "Classify disease/pathogenicity for variant: [REF][SEP][ALT][MASK]Label"
# Loss computed only on output after [MASK] (completion-only)
#
# =============================================================================

set -e

mkdir -p logs

cd /cluster/projects/bwanggroup/vsubasri/genomic-FM

source ~/miniconda3/etc/profile.d/conda.sh

conda activate genomic-fm

wandb offline

# =============================================================================
# Configuration
# =============================================================================

# Model (must be OmniDNA for generative training)
MODEL="omni_dna_116m"  # Options: omni_dna_116m, omni_dna_1b

# Training hyperparameters
LEARNING_RATE=1e-5
BATCH_SIZE=4  # Lower batch size for generative (more memory intensive)
NUM_EPOCHS=10
THRESHOLD=65

# NEFTune noise for regularization (0 to disable)
NEFTUNE_ALPHA=5.0

# =============================================================================

echo "============================================"
echo "Generative Multitask Training (SFTTrainer)"
echo "============================================"
echo "Model: $MODEL"
echo "Tasks: CLNDN + CLNSIG"
echo "NEFTune alpha: $NEFTUNE_ALPHA"
echo "============================================"

accelerate launch --config_file configs/ddp.yaml --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer_smart \
    --model "$MODEL" \
    --seed 127 \
    --generative \
    --clndn \
    --clnsig \
    --threshold $THRESHOLD \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --neftune_alpha $NEFTUNE_ALPHA \
    2>&1 | tee logs/smart_generative_${SLURM_JOB_ID}.log

echo ""
echo "============================================"
echo "Generative Training Completed!"
echo "============================================"
echo "Output: ./root/models/generative_${MODEL}_CLNDN_CLNSIG"
echo "============================================"
