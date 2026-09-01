#!/bin/bash
#SBATCH --job-name=clinvar_attack
#SBATCH --account=pmcc_ai_team_gpu
#SBATCH --partition=gpu_pmcc_ai_team
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
conda activate gvrep-b200
cd /cluster/projects/bwanggroup/vsubasri/genomic-FM

# Create logs directory
mkdir -p logs

# Get experiment ID from array task (default to 0 for local runs)
expid=${SLURM_ARRAY_TASK_ID:-0}
num_experiments=64

# ============================================
# CONFIGURATION (can be overridden via environment variables)
# ============================================
# Model: HuggingFace model name or local checkpoint path
# Base model (not trained on ClinVar):
# MODEL="./root/models/nucleotide-transformer-500m-human-ref"
# Pretrained on ClinVar CLNSIG (auto-loads best checkpoint):
MODEL="${MODEL:-./root/models/pretrain_model_nt_CLNSIG}"

# Grouping mode: gene, exon, cardiac_panel, cardiac_gene, hcm_gene
GROUPING="${GROUPING:-cardiac_gene}"

# Prediction target: CLNSIG (pathogenicity) or CLNDN (disease)
TARGET="${TARGET:-CLNSIG}"

# Attack mode: 0 = likelihood-based, 1 = embedding-based
USE_EMBEDDING="${USE_EMBEDDING:-0}"

# Training parameters
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
PATIENCE="${PATIENCE:-10}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"

# Data parameters
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
MIN_VARIANTS="${MIN_VARIANTS:-5}"
MAX_VARIANTS="${MAX_VARIANTS:-50}"
BALANCE_CLASSES="${BALANCE_CLASSES:-1}"

# Optional: disease subset file (for CLNDN target)
DISEASE_SUBSET_FILE="${DISEASE_SUBSET_FILE:-}"

# LiRA split mode: 0 = individual variants (default), 1 = entire groups in/out together (patient panel)
SPLIT_BY_GROUP="${SPLIT_BY_GROUP:-0}"

# ClinVar review status filtering (0-4 stars, default 1 = criteria provided)
MIN_REVIEW_STARS="${MIN_REVIEW_STARS:-1}"

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
echo "  Split by group: $SPLIT_BY_GROUP"
echo "  Min review stars: $MIN_REVIEW_STARS"
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
    --freeze_backbone $FREEZE_BACKBONE \
    --split_by_group $SPLIT_BY_GROUP \
    --min_review_stars $MIN_REVIEW_STARS"

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

# ============================================
# OTHER EXPERIMENTS TO RUN
# ============================================
# After completing the current experiment, run these additional configurations.
# Use --export to pass environment variables to sbatch:
#
# 1. CLNSIG with embedding-based attack (instead of likelihood):
#    sbatch --export=ALL,USE_EMBEDDING=1 slurm/run_clinvar_attack.sh all
#
# 2. CLNSIG with frozen backbone (head-only training):
#    sbatch --export=ALL,FREEZE_BACKBONE=1 slurm/run_clinvar_attack.sh all
#
# 3. CLNSIG on HCM genes subset:
#    sbatch --export=ALL,GROUPING=hcm_gene slurm/run_clinvar_attack.sh all
#
# 4. CLNDN (disease prediction) on cardiac genes:
#    sbatch --export=ALL,TARGET=CLNDN,MODEL=./root/models/pretrain_model_nt_CLNDN slurm/run_clinvar_attack.sh all
#
# 5. CLNDN on HCM genes:
#    sbatch --export=ALL,TARGET=CLNDN,MODEL=./root/models/pretrain_model_nt_CLNDN,GROUPING=hcm_gene slurm/run_clinvar_attack.sh all
#
# 6. Base model (not fine-tuned) - measures pre-training memorization:
#    sbatch --export=ALL,MODEL=./root/models/nt slurm/run_clinvar_attack.sh all
#
# 7. All genes (not just cardiac):
#    sbatch --export=ALL,GROUPING=gene slurm/run_clinvar_attack.sh all
#
# 8. Split by group (entire groups in/out together - patient panel scenario):
#    sbatch --export=ALL,SPLIT_BY_GROUP=1 slurm/run_clinvar_attack.sh all
#
# 9. No review status filtering (include all variants regardless of evidence):
#    sbatch --export=ALL,MIN_REVIEW_STARS=0 slurm/run_clinvar_attack.sh all
#
# For local/interactive runs (single experiment):
#    USE_EMBEDDING=1 bash slurm/run_clinvar_attack.sh all
#    SPLIT_BY_GROUP=1 bash slurm/run_clinvar_attack.sh all
#    TARGET=CLNDN MODEL=./root/models/pretrain_model_nt_CLNDN bash slurm/run_clinvar_attack.sh all
#
# AGGREGATION (run after all 64 experiments complete for each config):
#    python scripts/run_clinvar_attack.py --mode aggregate --target CLNSIG --grouping cardiac_gene \
#        --model ./root/models/pretrain_model_nt_CLNSIG --use_embedding 0 --freeze_backbone 0 \
#        --min_variants_per_gene 5 --max_variants_per_gene 50 --num_experiments 64
# ============================================
