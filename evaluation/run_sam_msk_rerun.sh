#!/bin/bash --login
# Rerun SAM ViT-B (original weights, no training) on MSK_shoulder with the paper's test pipeline
# (evaluation/test_vis.py -> utils/visual.get_test_record_from_ds), plus CHAOS_T1 as a control.
#
# Why: on the same 641 MSK_shoulder structures the published SAM ViT-B box Dice is 0.267, but the
# swapped-encoder evaluation gives 0.658; on the other 5 zero-shot datasets the two agree within 0.005.
# CHAOS_T1 is one of those 5 (published 0.924): if the rerun reproduces it, the pipeline is unchanged and
# the MSK_shoulder rerun is the number to trust.
#
# BEFORE submitting (once):
#   mkdir -p /scratch/user/s4670484/Eval_results/sam_msk_rerun/logs
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch evaluation/run_sam_msk_rerun.sh
#
# FIRST MINUTES - tail the .output log:
#   "device: AMD Instinct MI355X"
#   "Testing datasets: ['CHAOS_T1', 'MSK_shoulder']"
#   "Processing the dataset: .../CHAOS_T1/testing/" and a tqdm progress bar in the .error log
# At the end the summary prints the median box Dice per dataset next to the published value.
#SBATCH --job-name=sam-msk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/sam_msk_rerun/logs/slurm-sam-msk-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/sam_msk_rerun/logs/slurm-sam-msk-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/user/s4670484/SAMRI}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

ROOT=/scratch/user/s4670484
OUT=$ROOT/Eval_results/sam_msk_rerun/
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.hip); \
assert torch.cuda.is_available(), 'no GPU visible'; print('device:', torch.cuda.get_device_name(0))" || exit 1
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)"

python -m evaluation.test_vis \
    --test-image-path "$ROOT/Datasets/Zeroshot/" \
    --datasets CHAOS_T1 MSK_shoulder \
    --ckpt-path "$ROOT/pretrained_ckpt/sam_vit_b_01ec64.pth" \
    --save-path "$OUT" \
    --model-type vit_b \
    --device cuda || { echo "!!!!! FAILED: test_vis"; exit 1; }

echo; echo "===== summary: median box Dice (b_dice) over structures ====="
python - "$OUT/sam_vit_b_01ec64" <<'PY'
import pickle, sys
import numpy as np
published = {"CHAOS_T1": (0.924, 156), "MSK_shoulder": (0.267, 641)}  # SAM_vitb in all_results.csv
rec = pickle.load(open(sys.argv[1], "rb"))
for ds, rows in rec.items():
    d = np.array([v for r in rows for v in r["b_dice"]], float)
    pub, n_pub = published.get(ds, (float("nan"), 0))
    print(f"  {ds:14s} rerun {np.nanmedian(d):.3f} over {len(d)} structures | published {pub:.3f} over {n_pub}")
PY
echo; echo "finished: $(date)  -> $OUT/sam_vit_b_01ec64 (pickle, same format as the paper's test records)"
