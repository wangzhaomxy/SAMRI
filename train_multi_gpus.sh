#!/bin/bash --login
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=SAMRI
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:l40:3
#SBATCH --account=AccountString
#SBATCH -o slurm-%j.output
#SBATCH -e slurm-%j.error

module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate samri

python test.py

