#!/bin/bash
#SBATCH --account aip-wanglab
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --array=0-64
#SBATCH -t 10:00:00

# cd projects/aip-wanglab/fangcong

# source envs/dnabert2/bin/activate
# cd DNABERT2/

expid=$SLURM_ARRAY_TASK_ID
num_experiments=64

# random split
python reconstruct_genotype/guess_kolter_one_codon.py --expid $expid --num_experiments $num_experiments \
--use_delta 1 --mode head_only

# split by codon position/groups
python reconstruct_genotype/guess_kolter_one_codon.py --expid $expid --num_experiments $num_experiments \
--use_delta 1 --mode head_only --split_by_group 1


#python reconstruct_genotype/guess_kolter_one_codon.py --global_output_dir "/home/fangcong/scratch/checkpoints_12" --use_delta 1 --mode head_only \
#--batch_size 32 --epochs 200 --expid $expid --num_experiments $num_experiments --lr 0.05
#
