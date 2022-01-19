#!/bin/zsh
# Run via 'sbatch slurm_imagenet.sh' from this ('experiments') folder
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=a.artemenkov@skoltech.ru   # Where to send mail

#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-12:00:00
source ~/.zshrc
conda activate pytorch-env
srun python images.py --config-name=imagenet trainer=slurm trainer.gpus=1 trainer.num_nodes=1 $*
