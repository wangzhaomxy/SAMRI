#!/bin/bash --login
# What do the decoders predict where they score Dice ~0 (MSK_shoulder scapula cartilage, MSK_FLASH femoral
# cartilage)? 1 MI355X, ~10-20 min. Reuses the stored features and trained weights; trains nothing.
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_inspect_predictions.sh
# CHECK the .output log: "CHECK against ...: N instances matched, max |Dice difference| 0.000x" (the masks are the
# ones the evaluation scored), then the per-system table; figures in decoder_swap/inspect/png/.
#SBATCH --job-name=samri-inspect
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-inspect-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-inspect-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
python -c "import torch, matplotlib; assert torch.cuda.is_available(), 'no GPU visible'" \
    || { echo "samri-mi355x missing a package or no GPU (python: $(which python)); stopping"; exit 1; }
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/user/s4670484/SAMRI}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
ROOT=/scratch/user/s4670484
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"
python embedding_assess/inspect_predictions.py \
    --zeroshot-dir "$ROOT/Eval_results/embedding_research_zeroshot" \
    --knee-dir "$ROOT/Eval_results/embedding_research_knee" \
    --swap-dir "$ROOT/Eval_results/decoder_swap" --ckpt-dir "$ROOT/pretrained_ckpt" \
    --out-dir "$ROOT/Eval_results/decoder_swap/inspect" || { echo "!!!!! FAILED: inspect_predictions"; exit 1; }
echo "finished: $(date)"
