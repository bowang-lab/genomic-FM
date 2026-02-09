#!/bin/bash
# =============================================================================
# Continual Learning with LoRA + Replay + SLAO (2025 SOTA)
# =============================================================================
#
# Based on:
# - SLAO (2025): Single LoRA via orthogonal init + asymmetric merging
# - ACM Computing Surveys: Combine replay + regularization for best results
#
# Workflow (Biological Causality: Molecular → Clinical):
#   Task 1: MAVES - Molecular effect (DMS functional scores)
#   Task 2: CLNSIG - Pathogenicity (Benign vs Pathogenic)
#   Task 3: CLNDN - Disease type (which disease it causes)
#
# Rationale:
#   - Molecular effects (MAVES) are the foundation of variant impact
#   - Pathogenicity (CLNSIG) is determined by molecular dysfunction
#   - Disease specificity (CLNDN) is the downstream clinical manifestation
#
# Key features:
# - LoRA for ~1% trainable parameters
# - SLAO: Orthogonal initialization (new B orthogonal to old directions)
# - SLAO: Asymmetric A/B treatment (A preserved more, B more plastic)
# - SLAO: Time-aware scaling λ(t) = 1/√t
# - Surprise-based replay buffer
#
# =============================================================================

# Configuration
MODEL="nt"  # nucleotide-transformer
SEED=127
BATCH_SIZE=8
NUM_EPOCHS=10
LEARNING_RATE=1e-4  # Higher LR for LoRA
LORA_R=8
LORA_ALPHA=16
REPLAY_BUFFER_SIZE=1000
REPLAY_RATIO=0.2

# SLAO Configuration
SLAO_ALPHA_A=0.5        # A preservation factor (higher = more preserved)
SLAO_MERGE_FREQ=100     # Merge frequency during training

echo "=============================================="
echo "Continual Learning: LoRA + Replay + SLAO"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Task Order: MAVES → CLNSIG → CLNDN"
echo "  (Molecular → Pathogenicity → Disease)"
echo "LoRA rank: ${LORA_R}"
echo "Replay buffer size: ${REPLAY_BUFFER_SIZE}"
echo "SLAO α_A: ${SLAO_ALPHA_A}"
echo "=============================================="

# =============================================================================
# STEP 1: Train on MAVES (molecular effect - DMS functional scores)
#         Foundation: Learn how variants affect protein function
# =============================================================================
echo ""
echo "[Step 1/3] Training on MAVES (molecular effect)..."
echo "           Learning variant functional impact from DMS data"
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
    --replay_buffer_size ${REPLAY_BUFFER_SIZE}

MAVES_LORA="./root/models/model_${MODEL}_MAVES_score_lora_replay/lora_adapter"
MAVES_BUFFER="./root/models/replay_buffer_MAVES_score.pt"
echo "MAVES checkpoint: ${MAVES_LORA}"
echo "MAVES replay buffer: ${MAVES_BUFFER}"

# =============================================================================
# STEP 2: Train on CLNSIG (pathogenicity classification)
#         Build on molecular understanding to classify pathogenicity
# =============================================================================
echo ""
echo "[Step 2/3] Training on CLNSIG (pathogenicity) with SLAO..."
echo "           Loading LoRA from: ${MAVES_LORA}"
echo "           Loading replay from: ${MAVES_BUFFER}"
echo "           SLAO: Orthogonal B init + asymmetric merge (task 2)"
echo "           λ(2) = 1/√2 ≈ 0.71"
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
    --use_slao \
    --task_number 2 \
    --slao_alpha_A ${SLAO_ALPHA_A} \
    --slao_merge_frequency ${SLAO_MERGE_FREQ}

CLNSIG_LORA="./root/models/model_${MODEL}_CLNSIG_lora_replay/lora_adapter"
CLNSIG_BUFFER="./root/models/replay_buffer_CLNSIG.pt"
echo "CLNSIG checkpoint: ${CLNSIG_LORA}"
echo "CLNSIG replay buffer: ${CLNSIG_BUFFER}"

# =============================================================================
# STEP 3: Train on CLNDN (disease classification)
#         Most specific: which disease does this pathogenic variant cause?
# =============================================================================
echo ""
echo "[Step 3/3] Training on CLNDN (disease type) with SLAO..."
echo "           Loading LoRA from: ${CLNSIG_LORA}"
echo "           Loading replay from: ${CLNSIG_BUFFER}"
echo "           SLAO: Orthogonal B init + asymmetric merge (task 3)"
echo "           λ(3) = 1/√3 ≈ 0.58 (more conservative)"
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
    --use_slao \
    --task_number 3 \
    --slao_alpha_A ${SLAO_ALPHA_A} \
    --slao_merge_frequency ${SLAO_MERGE_FREQ}

CLNDN_LORA="./root/models/model_${MODEL}_CLNDN_lora_replay/lora_adapter"
CLNDN_BUFFER="./root/models/replay_buffer_CLNDN.pt"

echo ""
echo "=============================================="
echo "SLAO Continual Learning Pipeline Complete!"
echo "=============================================="
echo ""
echo "Task Order (Biological Causality):"
echo "  1. MAVES  - Molecular effect (foundation)"
echo "  2. CLNSIG - Pathogenicity (clinical significance)"
echo "  3. CLNDN  - Disease type (specific diagnosis)"
echo ""
echo "SLAO applied:"
echo "  - Orthogonal B initialization (avoids overwriting previous knowledge)"
echo "  - Asymmetric A/B merging (A preserved, B plastic)"
echo "  - Time-aware scaling: λ(2)=0.71, λ(3)=0.58"
echo ""
echo "Checkpoints created:"
echo "  1. ${MAVES_LORA} (MAVES only)"
echo "  2. ${CLNSIG_LORA} (MAVES + CLNSIG via SLAO)"
echo "  3. ${CLNDN_LORA} (MAVES + CLNSIG + CLNDN via SLAO)"
echo ""
echo "Replay buffers (for future tasks):"
echo "  - ${MAVES_BUFFER}"
echo "  - ${CLNSIG_BUFFER}"
echo "  - ${CLNDN_BUFFER}"
echo ""
echo "To evaluate on all tasks with final model:"
echo "  python -m src.pack_tunable_model.hf_trainer_continual --model ${MODEL} --task MAVES --test_only --use_lora --lora_checkpoint ${CLNDN_LORA}"
echo ""
