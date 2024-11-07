#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=SAMRI
#SBATCH --time=96:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40:3
#SBATCH --account=a_auto_mr_disease
#SBATCH --qos=gpu
#SBATCH -o /home/s4670484/Documents/slurm-%j.output
#SBATCH -e /home/s4670484/Documents/slurm-%j.error

module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri

python train_fast.py

