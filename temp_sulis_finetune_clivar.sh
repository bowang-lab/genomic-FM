#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=35
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su114


module purge
#module load GCC/10.2.0 CUDA/11.3.1  OpenMPI/4.0.5 Python/3.8.6
#source ~/torch/bin/activate
module load CUDA/11.7.0 GCCcore/11.3.0 GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4
source ~/torch_1_13_1/bin/activate
cd ~/repositories/genomic-FM

srun python test_finetune.py --dataset='clinvar_CLNSIG_hyena-tiny' --epochs=100 --gpus=1 --num_workers=24
# accelerate launch --main_process_port 12701 --multi_gpu main.py --model vanilla_vae
# accelerate launch --main_process_port 12701 --multi_gpu main.py --model un_unet
# accelerate launch --main_process_port 12902 --multi_gpu main.py --model un_unet
# accelerate launch --main_process_port 12903 --multi_gpu main.py --model generate --gen_vae_path=/home/z/zehuiml/repositories/dna-diffusion/saved_models/vae_best_model.pth --gen_unet_path=/home/z/zehuiml/repositories/dna-diffusion/saved_models/2023-09-20_14-19-38_fid_test_0.pth
# accelerate launch --main_process_port 12903 --multi_gpu main.py --model generate --gen_vae_path=/home/z/zehuiml/repositories/dna-diffusion/saved_models/vae_best_model.pth --gen_unet_path=/home/z/zehuiml/repositories/dna-diffusion/saved_models/2023-09-20_14-19-38_fid_test_50.pth
# accelerate launch --main_process_port 12903 --multi_gpu main.py --model generate --gen_vae_path=/home/z/zehuiml/repositories/dna-diffusion/saved_models/vae_best_model.pth --gen_unet_path=/home/z/zehuiml/repositories/dna-diffusion/saved_models/2023-09-20_14-19-38_fid_test_100.pth
# accelerate launch --main_process_port 12903 --multi_gpu main.py --model generate --gen_vae_path=/home/z/zehuiml/repositories/dna-diffusion/saved_models/vae_best_model.pth --gen_unet_path=/home/z/zehuiml/repositories/dna-diffusion/saved_models/2023-09-20_14-19-38_fid_test_300.pth
# CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 12905 --multi_gpu main.py --model generate --gen_vae_path=/home/zl6222/repositories/dna-diffusion/saved_models/vae_9939_2023-09-21_15-31-56_best_model.pth --gen_unet_path=/home/zl6222/repositories/dna-diffusion/saved_models/2023-09-24_23-57-45_score_func_1000.pth
# CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 12905 --multi_gpu main.py --model generate --gen_vae_path=/home/zl6222/repositories/dna-diffusion/saved_models/2023-09-28_19-51-45baseline_best_model.pth --gen_unet_path=/home/zl6222/repositories/dna-diffusion/saved_models/2023-09-24_23-57-45_score_func_1000.pth
# CUDA_VISIBLE_DEVICES="2" accelerate launch --main_process_port 12904 --multi_gpu main.py --model generate --gen_vae_path=/home/zl6222/repositories/dna-diffusion/saved_models/vae_9939_2023-09-21_15-31-56_best_model.pth --gen_unet_path=/home/zl6222/repositories/dna-diffusion/saved_models/2023-09-24_23-57-45_score_func_2000.pth
# CUDA_VISIBLE_DEVICES="1" accelerate launch --main_process_port 12903 --multi_gpu main.py --model generate --gen_vae_path=/home/zl6222/repositories/dna-diffusion/saved_models/vae_9939_2023-09-21_15-31-56_best_model.pth --gen_unet_path=/home/zl6222/repositories/dna-diffusion/saved_models/2023-09-24_23-57-45_score_func_3000.pth
# accelerate launch --main_process_port 12701 --multi_gpu main.py --model conditional_generate --gen_vae_path=/home/z/zehuiml/repositories/icml-dna-diffusion/dna-diffusion/saved_models/icml_central256_2024-01-05_06-49-32baseline_best_model.pth --gen_unet_path=/home/z/zehuiml/repositories/icml-dna-diffusion/dna-diffusion/saved_models/2024-01-11_01-14-57_score_func_400.pth
# CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 12701 --multi_gpu main.py --model conditional_generate --gen_vae_path=/home/z/zehuiml/repositories/icml-dna-diffusion/dna-diffusion/saved_models/icml_central256_2024-01-05_06-49-32baseline_best_model.pth --gen_unet_path=/home/z/zehuiml/repositories/icml-dna-diffusion/dna-diffusion/saved_models/2024-01-11_01-14-57_score_func_200.pth
# CUDA_VISIBLE_DEVICES="1" accelerate launch --main_process_port 12701 --multi_gpu main.py --model conditional_generate --gen_vae_path=/home/z/zehuiml/repositories/icml-dna-diffusion/dna-diffusion/saved_models/icml_central256_2024-01-05_06-49-32baseline_best_model.pth --gen_unet_path=/home/z/zehuiml/repositories/icml-dna-diffusion/dna-diffusion/saved_models/2024-01-11_01-14-57_score_func_600.pth
