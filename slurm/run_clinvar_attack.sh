#!/bin/bash
#SBATCH --job-name=clinvar_attack
#SBATCH --account=bwanggroup_gpu
#SBATCH --partition=gpu_short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-63
#SBATCH --output=logs/attack_%A_%a.out
#SBATCH --error=logs/attack_%A_%a.err

set -e

# ============================================
# USAGE:
#   sbatch slurm/run_clinvar_attack.sh train    # Train shadow models
#   sbatch slurm/run_clinvar_attack.sh eval     # Evaluate + run attack
#   sbatch slurm/run_clinvar_attack.sh all      # Train + eval in one job
#
# AGGREGATE (after all jobs complete):
#   python scripts/run_clinvar_attack.py --mode aggregate \
#       --target CLNSIG --grouping cardiac_gene \
#       --model ./root/models/pretrain_model_nt_CLNSIG \
#       --use_embedding 0 --freeze_backbone 0 \
#       --min_variants_per_gene 5 --max_variants_per_gene 50 \
#       --num_experiments 64
#
# OUTPUT: attack_results/{TARGET}_{GROUPING}_{MODEL}_{lik|emb}_{head|full}[_v{MIN}-{MAX}][_{SUBSET}]/
# ============================================

# Get mode from command line argument (default: all)
MODE=${1:-all}

# Environment setup
source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomic-fm
cd /cluster/projects/bwanggroup/vsubasri/genomic-FM

# Create logs directory
mkdir -p logs

# Get experiment ID from array task
expid=$SLURM_ARRAY_TASK_ID
num_experiments=64

# ============================================
# CONFIGURATION
# ============================================
# Model: HuggingFace model name or local checkpoint path
# Base model (not trained on ClinVar):
# MODEL="./root/models/nucleotide-transformer-500m-human-ref"
# Pretrained on ClinVar CLNSIG (auto-loads best checkpoint):
MODEL="./root/models/pretrain_model_nt_CLNSIG"

# Grouping mode: gene, exon, cardiac_panel, cardiac_gene, hcm_gene
GROUPING="cardiac_gene"

# Prediction target: CLNSIG (pathogenicity) or CLNDN (disease)
TARGET="CLNSIG"

# Attack mode: 0 = likelihood-based, 1 = embedding-based
USE_EMBEDDING=0

# Training parameters
EPOCHS=50
BATCH_SIZE=16
LR=1e-4
PATIENCE=10
FREEZE_BACKBONE=0

# Data parameters
SEQ_LENGTH=1024
MIN_VARIANTS=5
MAX_VARIANTS=50
BALANCE_CLASSES=1

# Optional: disease subset file (for CLNDN target)
DISEASE_SUBSET_FILE=""

# Job information
echo "===== ClinVar Attribute Inference Attack ====="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "=============================================="
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Experiment: $expid / $num_experiments"
echo "  Target: $TARGET"
echo "  Grouping: $GROUPING"
echo "  Model: $MODEL"
echo "  Seq length: $SEQ_LENGTH"
echo "  Min/Max variants: $MIN_VARIANTS / $MAX_VARIANTS"
echo "  Balance classes: $BALANCE_CLASSES"
echo "  Use embedding: $USE_EMBEDDING"
if [ "$MODE" = "train" ]; then
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Learning rate: $LR"
    echo "  Freeze backbone: $FREEZE_BACKBONE"
fi
echo "=============================================="

# Base command arguments
BASE_ARGS="--expid $expid \
    --num_experiments $num_experiments \
    --grouping $GROUPING \
    --target $TARGET \
    --model $MODEL \
    --seq_length $SEQ_LENGTH \
    --min_variants_per_gene $MIN_VARIANTS \
    --max_variants_per_gene $MAX_VARIANTS \
    --balance_classes $BALANCE_CLASSES \
    --use_embedding $USE_EMBEDDING \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --patience $PATIENCE \
    --freeze_backbone $FREEZE_BACKBONE"

# Add disease subset file if specified
if [ -n "$DISEASE_SUBSET_FILE" ]; then
    BASE_ARGS="$BASE_ARGS --disease_subset_file $DISEASE_SUBSET_FILE"
fi

# Run based on mode
if [ "$MODE" = "all" ]; then
    # Run train then eval
    echo "=== STEP 1: Training ==="
    CMD="python scripts/run_clinvar_attack.py --mode train $BASE_ARGS"
    echo "Running: $CMD"
    eval $CMD

    echo ""
    echo "=== STEP 2: Evaluation ==="
    CMD="python scripts/run_clinvar_attack.py --mode eval $BASE_ARGS"
    echo "Running: $CMD"
    eval $CMD
else
    # Run single mode (train or eval)
    CMD="python scripts/run_clinvar_attack.py --mode $MODE $BASE_ARGS"
    echo "Running: $CMD"
    eval $CMD
fi

EXIT_CODE=$?

echo ""
echo "===== JOB SUMMARY ====="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
[ $EXIT_CODE -eq 0 ] && echo "Attack completed successfully!" || echo "Attack failed"
echo "========================"

exit $EXIT_CODE
