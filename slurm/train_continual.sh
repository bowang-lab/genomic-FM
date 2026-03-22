#!/bin/bash
#SBATCH -t 4-00:0:0
#SBATCH -J train_continual
#SBATCH -p gpu_bwanggroup
#SBATCH --account=bwanggroup_gpu
#SBATCH --mem=450G
#SBATCH -c 8
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --output=logs/continual_output_%j.log
#SBATCH --error=logs/continual_error_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vallisubasri@gmail.com

# =============================================================================
# Continual Learning with LoRA + Replay + SLAO
# =============================================================================
#
# Task Order (Biological Causality: Molecular → Clinical):
#   Task 1: MAVES - Molecular effect (DMS functional scores)
#   Task 2: CLNSIG - Pathogenicity (Benign vs Pathogenic)
#   Task 3: CLNDN - Disease type (which disease it causes)
#
# Key features:
# - LoRA for parameter-efficient fine-tuning (~1% trainable params)
# - Replay buffer to prevent catastrophic forgetting
# - SLAO: Orthogonal B initialization + asymmetric merging (tasks 2+3)
#
# =============================================================================

set -e

mkdir -p logs

cd /cluster/projects/bwanggroup/vsubasri/genomic-FM

source ~/miniconda3/etc/profile.d/conda.sh

conda activate genomic-fm

wandb offline

# Configuration
MODEL="nt"  # Options: nt, omni_dna_116m, dnabert2, hyenadna, etc.
SEED=127
BATCH_SIZE=8
NUM_EPOCHS=10
LEARNING_RATE=1e-4
LORA_R=8
LORA_ALPHA=16

# Replay config (based on Kotha & Liang 2026: higher ratios improve target performance)
REPLAY_BUFFER_SIZE=5000
REPLAY_RATIO=0.5  # 0.5 = equal mix of current and replay samples

# Set decoder flag for autoregressive models
if [[ "$MODEL" == "hyenadna" || "$MODEL" == "omni_dna_116m" ]]; then
    DECODER_FLAG="--decoder"
else
    DECODER_FLAG=""
fi

echo "=============================================="
echo "Continual Learning: LoRA + Replay + SLAO"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Task Order: MAVES → CLNSIG → CLNDN"
echo "LoRA rank: ${LORA_R}"
echo "Replay: buffer=${REPLAY_BUFFER_SIZE}, ratio=${REPLAY_RATIO} (data-level mixing)"
echo "SLAO: Enabled for tasks 2 and 3"
echo "=============================================="

# =============================================================================
# STEP 1: Train on MAVES (molecular effect - DMS functional scores)
# =============================================================================
echo ""
echo "[Step 1/3] Training on MAVES (molecular effect)..."
echo ""

accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer_continual \
    --model ${MODEL} \
    --task MAVES \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --experimental_methods DMS \
    --region_type coding \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --use_replay \
    --replay_buffer_size ${REPLAY_BUFFER_SIZE} \
    $DECODER_FLAG

MAVES_LORA="./root/models/continual_${MODEL}_MAVES_score_lora_replay/lora_adapter"
MAVES_BUFFER="./root/models/replay_buffer_MAVES_score.pt"
echo "MAVES checkpoint: ${MAVES_LORA}"

# =============================================================================
# STEP 2: Train on CLNSIG (pathogenicity classification)
# =============================================================================
echo ""
echo "[Step 2/3] Training on CLNSIG (pathogenicity) with SLAO..."
echo ""

accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer_continual \
    --model ${MODEL} \
    --task CLNSIG \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_checkpoint ${MAVES_LORA} \
    --use_replay \
    --replay_buffer ${MAVES_BUFFER} \
    --replay_buffer_size ${REPLAY_BUFFER_SIZE} \
    --replay_ratio ${REPLAY_RATIO} \
    --slao --task_number 2 \
    $DECODER_FLAG

CLNSIG_LORA="./root/models/continual_${MODEL}_CLNSIG_lora_replay/lora_adapter"
CLNSIG_BUFFER="./root/models/replay_buffer_CLNSIG.pt"
echo "CLNSIG checkpoint: ${CLNSIG_LORA}"

# =============================================================================
# STEP 3: Train on CLNDN (disease classification)
# =============================================================================
echo ""
echo "[Step 3/3] Training on CLNDN (disease type) with SLAO..."
echo ""

accelerate launch \
    --config_file configs/ddp.yaml \
    --main_process_port 29500 \
    -m src.pack_tunable_model.hf_trainer_continual \
    --model ${MODEL} \
    --task CLNDN \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --use_lora \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_checkpoint ${CLNSIG_LORA} \
    --use_replay \
    --replay_buffer ${CLNSIG_BUFFER} \
    --replay_buffer_size ${REPLAY_BUFFER_SIZE} \
    --replay_ratio ${REPLAY_RATIO} \
    --slao --task_number 3 \
    $DECODER_FLAG

CLNDN_LORA="./root/models/continual_${MODEL}_CLNDN_lora_replay/lora_adapter"

echo ""
echo "=============================================="
echo "Continual Learning Complete!"
echo "=============================================="
echo ""
echo "Checkpoints:"
echo "  1. ${MAVES_LORA}"
echo "  2. ${CLNSIG_LORA}"
echo "  3. ${CLNDN_LORA}"
echo ""
echo "To evaluate:"
echo "  python -m src.pack_tunable_model.hf_trainer_continual --model ${MODEL} --task MAVES --test_only --use_lora --lora_checkpoint ${CLNDN_LORA}"
echo ""
