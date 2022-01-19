#!/bin/zsh
# Run via 'sbatch slurm_imagenet.sh' from this ('experiments') folder
#SBATCH --mail-type=END,FAIL                 # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=a.artemenkov@skoltech.ru   # Where to send mail

#SBATCH --partition=gpu
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=6-00:00:00
source ~/.zshrc
conda activate pytorch-env
srun python images.py --config-name=imagenet trainer=slurm trainer.gpus=2 trainer.num_nodes=3 $*
