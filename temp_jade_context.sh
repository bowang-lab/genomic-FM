#!/bin/bash
# set the number of nodes
#SBATCH --nodes=1
# set max wallclock time
#SBATCH --time=36:00:00
# set name of job
#SBATCH --job-name=jobzl432
# set number of GPUs
#SBATCH --gres=gpu:1 -p small
# set RAM size
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=zl432@cantab.ac.uk
module load pytorch/1.12.1
source activate epi
cd /jmain02/home/J2AD015/axf03/zxl79-axf03/repository/genomic-FM-run-exp/genomic-FM
wandb offline

####################
# 512 sqtl_pval
####################
python test_finetune_delta.py --dataset='clinvar_CLNDN-512_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-512_dnabert2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-512_nt' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-512_ntv2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0

####################
# 256 clinvar_CLNDN
####################

python test_finetune_delta.py --dataset='clinvar_CLNDN-256_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-256_dnabert2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-256_nt' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml  --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-256_ntv2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0

####################
# 128 clinvar_CLNDN
####################

python test_finetune_delta.py --dataset='clinvar_CLNDN-128_hyena-tiny' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-128_dnabert2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-128_nt' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
python test_finetune_delta.py --dataset='clinvar_CLNDN-128_ntv2' --epochs=100 --gpus=1 --num_workers=8 --config=configs/finetune_var_context.yaml --seed=0
