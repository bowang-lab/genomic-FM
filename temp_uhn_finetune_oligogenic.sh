#!/bin/bash
#SBATCH --nodes=1
#SBATCH -t 3-12:0:0
#SBATCH -J oligogenic
#SBATCH -p gpu_bwanggroup
#SBATCH --gres=gpu:1 
#SBATCH --mem=50G
#SBATCH -c 20
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vallisubasri@gmail.com
#SBATCH --output=root/oligogenic_output_%j.log 
#SBATCH --error=root/oligogenic_error_%j.log  

# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"=$SLURM_SUBMIT_DIR

module load git/2.42.0

cd /cluster/projects/aihub/vsubasri/genomic-FM

source ~/miniconda3/etc/profile.d/conda.sh
conda activate genomic-fm
wandb offline

python test_finetune_delta.py --dataset='oligogenic_hyena-tiny' --epochs=100 --gpus=1 --num_workers=16

