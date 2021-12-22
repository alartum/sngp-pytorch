#!/bin/zsh
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=04:00:00
source ~/.zshrc
conda activate pytorch-env
srun python images.py model=resnet50 data=cifar100 trainer=slurm
