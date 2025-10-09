#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1T
#SBATCH --job-name=SAMRI
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:8
#SBATCH --account=a_auto_mr_disease
#SBATCH --qos=sdf
#SBATCH -o /home/s4670484/Documents/slurm-%j.output
#SBATCH -e /home/s4670484/Documents/slurm-%j.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri-mi300

# Dynamically assign port from job ID to avoid collisions
export MASTER_ADDR=localhost
export MASTER_PORT=$((26000 + RANDOM % 1000))  # Pick a port between 26000 ~ 26999

amd-smi monitor
python train_multi_gpus.py

