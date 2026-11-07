#!/bin/bash --login
# Decoder-only data-size and data-variety runs (frozen SAM ViT-B encoder), 1 GPU (MI355X, --qos=gpu).
#
#   data size     decoder trained / LoRA on decoder, 15 50 150 500 1500 5000 slices per dataset
#                 (equal per dataset, so BraTS no longer dominates), seeds 0 1 2
#   data variety  decoder trained on 1, 3, 6 or 11 of the 11 datasets, 1,500 slices in total, 3 draws each;
#                 the left-out clean datasets are extra unseen test sets
# The slice pool (<= 5,000 per dataset) is encoded once (~15-30 min) and kept in GPU memory; every run then
# trains in minutes. Same recipe, validation slices and test sets as the swapped-encoder arms.
#
# Can run AT THE SAME TIME as run_lora_experiments.sh (separate GPU, separate output folder). It needs the
# decoder-LoRA learning rate that job chose in its step 1 (read from decoder_swap/runs/
# sam-enc_sam-dec-lora_small/seed0/meta.json); or give it:  sbatch --export=ALL,DEC_LORA_LR=5e-4 ...
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_curve_experiments.sh
# Finished runs (DONE) are skipped on resubmission; only the pool encoding is repeated.
#
# FIRST MINUTES - tail the .output log:
#   "device: AMD Instinct MI355X"
#   "pool <dataset>: N slices" for 11 datasets, then "pool: ... slices (... GB on GPU once encoded)"
#   "[check] encoder output vs stored features ... cosine min ~1.0000"
#   "encoding batch 1/..." with a time-left estimate
# AFTER ENCODING: "sam-enc_sam-dec_n15 seed 0: ... validation Dice before training 0.795" (untrained SAM decoder
#   on the shared validation slices, as in every earlier arm), then epochs of a second or two.
#SBATCH --job-name=samri-curve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-curve-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-curve-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
python -c "import torch, torchvision, numpy, nibabel, skimage, scipy.stats" \
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
INDOMAIN=$ROOT/Eval_results/embedding_research_clean
ZEROSHOT=$ROOT/Eval_results/embedding_research_zeroshot
OUT=$ROOT/Eval_results/decoder_swap
RUNS=$OUT/runs
CURVE=$OUT/runs_curve
CKPT=$ROOT/pretrained_ckpt
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
for f in "$INDOMAIN/features/sam_vit_b__sam/DONE" "$ZEROSHOT/features/sam_vit_b__sam/DONE" "$RUNS/val_rows.json"; do
    [ -f "$f" ] || { echo "Missing $f - run run_decoder_experiments.sh first"; exit 1; }
done
DEC_LR=$(cat "$OUT/chosen_lr.txt" 2>/dev/null || echo 1e-4)       # the decoder arms' learning rate
if [ -z "${DEC_LORA_LR:-}" ]; then
    DEC_LORA_LR=$(python -c "import json; print(json.load(open('$RUNS/sam-enc_sam-dec-lora_small/seed0/meta.json'))['lr'])" \
        2>/dev/null) || { echo "No decoder-LoRA learning rate yet (run_lora_experiments.sh step 1) - give DEC_LORA_LR"; exit 1; }
fi
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"
echo "learning rates: decoder $DEC_LR, decoder LoRA $DEC_LORA_LR"

# ---- 1. all runs (one process: the pool is encoded once) -----------------------------------------
python embedding_assess/train_decoder_curve.py --work-dir "$INDOMAIN" --val-rows "$RUNS/val_rows.json" \
    --ckpt-dir "$CKPT" --out-dir "$CURVE" --workers "$WORKERS" \
    --lr-decoder "$DEC_LR" --lr-decoder-lora "$DEC_LORA_LR" \
    || echo "!!!!! FAILED: train_decoder_curve (finished runs are kept; resubmit to continue)"

# ---- 2. evaluation (stored features, decoders only) -------------------------------------------------
echo; echo "===== evaluation ====="
python embedding_assess/eval_decoder.py --indomain-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" \
    --runs-dir "$CURVE" --ckpt-dir "$CKPT" --out-dir "$OUT/results_curve" \
    || echo "!!!!! FAILED: evaluation"

# ---- 3. tables ------------------------------------------------------------------------------------
echo; echo "===== tables ====="
python embedding_assess/analyse_curve.py --instances "$OUT/results_curve/eval_instances.csv" \
    --runs "$CURVE" --pool-json "$CURVE/pool.json" --out-dir "$OUT/results_curve/tables" \
    || echo "!!!!! FAILED: tables"
echo; echo "all steps finished: $(date)"
