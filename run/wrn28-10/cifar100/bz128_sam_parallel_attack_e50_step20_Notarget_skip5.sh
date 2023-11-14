#!/bin/bash

#SBATCH --job-name=1sam_p_50e

#SBATCH --output=/lustre/scratch/client/vinai/users/trunglm12/OT-MDR/log_files/cifar100/wrn28-10/bz128_sam_parallel_attack_e50_step20_Notarget_skip5.out

#SBATCH --error=/lustre/scratch/client/vinai/users/trunglm12/OT-MDR/log_files/cifar100/wrn28-10/bz128_sam_parallel_attack_e50_step20_Notarget_skip5.err

#SBATCH --gpus=1

#SBATCH --nodes=1

#SBATCH --mem-per-gpu=40G

#SBATCH --cpus-per-gpu=32

#SBATCH --partition=research
#SBATCH --mail-type=all #SBATCH --mail-user=trunglm12@vinai.io

module purge

eval "$(conda shell.bash hook)"

source activate /home/trunglm12/.conda/envs/vit
export PYTHONPATH=$PWD

python train_DataModelRo.py --rho 0.1 --dataset_path ../vit_selfOT/ViT-pytorch/data --dataset cifar100 \
 --epochs 200 --batch_size 128 --sam --lr_schedule cosine --attack p --atk_epoch 50 --atk_step 20 --atk_skip 5 \
 --log_dir log_files/cifar100/wrn28-10/bz128_sam_parallel_attack_e50_step20_Notarget_skip5