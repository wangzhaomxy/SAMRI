#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=SAMRI
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:2
#SBATCH --account=a_auto_mr_disease
#SBATCH --qos=gpu
#SBATCH -o /home/s4670484/Documents/slurm-%j.output
#SBATCH -e /home/s4670484/Documents/slurm-%j.error

module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri-rocm

nvidia-smi
python train_in_batch_multi_gpu.py

