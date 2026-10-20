#!/bin/bash --login
# Rerun the MedSAM baseline with the fixed input preprocessing (evaluation/test_medsam.py, fixed 14 Oct 2026:
# MedSAM's own MR percentile clip, no uint8 wrap-around), then score it (evaluation/test_medsam_eval.py).
#
# Default datasets: the 18 whose published MedSAM input was corrupted (medsam_check.py on the whole test set):
#   >= 79% of slices: Brain_Tumor_Dataset_Figshare MSD_Heart MSD_Prostate MSK_FLASH PROMISE QUBIQ_kidney
#                     MSD_BrainTumour Picai MSD_Hippocampus  + Zeroshot: CHAOS_T2 QUBIQ_prostate
#   partly:           CHAOS_T1 OAIAKOA HipMRI ZIB_OAI MSK_PD MSK_T2 ACDC
# All 36 datasets instead (recommended for a consistent baseline - the fix also changes the input of
# uncorrupted bright-tailed datasets such as MSK_shoulder):
#   sbatch --time=12:00:00 --export=ALL,DATASETS=all evaluation/run_medsam_rerun.sh
#
# BEFORE submitting (once):
#   mkdir -p /scratch/user/s4670484/Eval_results/medsam_rerun/logs
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch evaluation/run_medsam_rerun.sh
# Each dataset writes <OUT>/inference/<dataset>/DONE and is skipped on resubmission, so a cut job loses
# at most one dataset; resubmit the same command.
#
# FIRST MINUTES - tail the .output log:
#   "device: AMD Instinct MI355X"
#   "===== <first dataset> =====" then "Testing datasets: ['<first dataset>']" and "Processing the dataset: ..."
#   files appearing in /scratch/user/s4670484/Eval_results/medsam_rerun/inference/<dataset>/results/
#   no repeated "Error in file:" lines (a few on empty/odd slices are expected, hundreds are not)
#SBATCH --job-name=medsam-rerun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/medsam_rerun/logs/slurm-medsam-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/medsam_rerun/logs/slurm-medsam-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
python -c "import torch, numpy, nibabel, skimage, scipy" \
    || { echo "samri-mi355x is missing a package (python: $(which python)); stopping"; exit 1; }
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/user/s4670484/SAMRI}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

ROOT=/scratch/user/s4670484
OUT=$ROOT/Eval_results/medsam_rerun
INFER=$OUT/inference/
ROOTS="$ROOT/Datasets/SAMRI_train_test/ $ROOT/Datasets/Zeroshot/"
AFFECTED="Brain_Tumor_Dataset_Figshare MSD_Heart MSD_Prostate MSK_FLASH PROMISE QUBIQ_kidney MSD_BrainTumour Picai MSD_Hippocampus CHAOS_T2 QUBIQ_prostate CHAOS_T1 OAIAKOA HipMRI ZIB_OAI MSK_PD MSK_T2 ACDC"
DATASETS=${DATASETS:-$AFFECTED}
if [ "$DATASETS" = all ]; then
    DATASETS=$(for r in $ROOTS; do for d in "$r"*/; do [ -d "$d/testing" ] && basename "$d"; done; done)
fi
mkdir -p "$INFER"
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.hip); \
assert torch.cuda.is_available(), 'no GPU visible'; print('device:', torch.cuda.get_device_name(0))" || exit 1
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)"
echo "datasets ($(echo $DATASETS | wc -w)): $DATASETS"

# ---- 1. inference, one dataset per call (resumable) ----------------------------------------
for ds in $DATASETS; do
    if [ -f "$INFER/$ds/DONE" ]; then echo "$ds: done - skipping"; continue; fi
    echo; echo "===== $ds ($(date +%H:%M)) ====="
    python -m evaluation.test_medsam --test-image-path $ROOTS --datasets "$ds" \
        --ckpt-path "$ROOT/pretrained_ckpt/medsam_vit_b.pth" --save-path "$INFER" --no-comb --device cuda \
        && { mkdir -p "$INFER/$ds"; touch "$INFER/$ds/DONE"; } \
        || echo "!!!!! FAILED: $ds"
done

# ---- 2. scoring (Dice, HD, MSD per structure) ------------------------------------------------
echo; echo "===== scoring ====="
python -m evaluation.test_medsam_eval --medsam-infer-path "$INFER" --save-path "$OUT/medsam_fixed_records.pkl" \
    || { echo "!!!!! FAILED: scoring"; exit 1; }

# ---- 3. summary: median box Dice per dataset ------------------------------------------------
python - "$OUT/medsam_fixed_records.pkl" <<'PY'
import pickle, sys
import numpy as np
rec = pickle.load(open(sys.argv[1], "rb"))
print("\nMedSAM (fixed input) - median box Dice per dataset")
for ds in sorted(rec):
    d = np.array([v for r in rec[ds] for v in np.atleast_1d(r["b_dice"])], float)
    print(f"  {ds:30s} {np.nanmedian(d):.3f}  ({len(d)} structures)")
PY
echo; echo "finished: $(date)  -> $OUT/medsam_fixed_records.pkl"
