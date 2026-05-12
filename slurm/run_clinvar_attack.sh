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
CHECKPOINT="./root/models/smart_pretrain_model_nt_CLNSIG_threshold_65_with_state_dict"
MODEL_TYPE="nt"
GROUPING="gene"  # Options: gene, exon, cardiac_panel, cardiac_gene, hcm_gene

# Job information
echo "===== ClinVar Attribute Inference Attack ====="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "=============================================="
echo "Configuration:"
echo "  Experiment: $expid / $num_experiments"
echo "  Grouping: $GROUPING"
echo "  Checkpoint: $CHECKPOINT"
echo "  Model type: $MODEL_TYPE"
echo "=============================================="

# Run attack
python scripts/run_clinvar_attack.py \
    --expid $expid \
    --num_experiments $num_experiments \
    --grouping $GROUPING \
    --checkpoint "$CHECKPOINT" \
    --model_type $MODEL_TYPE \
    --eval_only 1

EXIT_CODE=$?

echo ""
echo "===== JOB SUMMARY ====="
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
[ $EXIT_CODE -eq 0 ] && echo "Attack completed successfully!" || echo "Attack failed"
echo "========================"

exit $EXIT_CODE
