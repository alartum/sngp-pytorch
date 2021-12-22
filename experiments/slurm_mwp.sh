#!/bin/zsh
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=5G
#SBATCH --time=04:00:00
source ~/.zshrc
conda activate pytorch-env
srun python mwp.py model=large data=imagenet trainer=slurm
