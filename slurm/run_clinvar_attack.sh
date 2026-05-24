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
MODEL="InstaDeepAI/nucleotide-transformer-500m-human-ref"

# Grouping mode: gene, exon, cardiac_panel, cardiac_gene, hcm_gene
GROUPING="cardiac_gene"

# Prediction target: CLNSIG (pathogenicity) or CLNDN (disease)
TARGET="CLNSIG"

# Attack mode: 0 = likelihood-based, 1 = embedding-based
USE_EMBEDDING=0

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
echo "  Experiment: $expid / $num_experiments"
echo "  Target: $TARGET"
echo "  Grouping: $GROUPING"
echo "  Model: $MODEL"
echo "  Seq length: $SEQ_LENGTH"
echo "  Min/Max variants: $MIN_VARIANTS / $MAX_VARIANTS"
echo "  Balance classes: $BALANCE_CLASSES"
echo "  Use embedding: $USE_EMBEDDING"
echo "=============================================="

# Build command
CMD="python scripts/run_clinvar_attack.py \
    --expid $expid \
    --num_experiments $num_experiments \
    --grouping $GROUPING \
    --target $TARGET \
    --model $MODEL \
    --seq_length $SEQ_LENGTH \
    --min_variants_per_gene $MIN_VARIANTS \
    --max_variants_per_gene $MAX_VARIANTS \
    --balance_classes $BALANCE_CLASSES \
    --use_embedding $USE_EMBEDDING \
    --eval_only 1"

# Add disease subset file if specified
if [ -n "$DISEASE_SUBSET_FILE" ]; then
    CMD="$CMD --disease_subset_file $DISEASE_SUBSET_FILE"
fi

# Run attack
echo "Running: $CMD"
eval $CMD

EXIT_CODE=$?

echo ""
echo "===== JOB SUMMARY ====="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
[ $EXIT_CODE -eq 0 ] && echo "Attack completed successfully!" || echo "Attack failed"
echo "========================"

exit $EXIT_CODE
