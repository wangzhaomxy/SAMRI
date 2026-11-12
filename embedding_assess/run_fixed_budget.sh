#!/bin/bash --login
# Data size at a FIXED training budget, 1 GPU (MI355X, --qos=gpu), ~3 h.
#
# The original data-size runs (run_curve_experiments.sh) stop early on the shared validation slices, so a bigger
# data set also trains longer (best epoch after ~1-4k training samples with 15 slices per dataset, ~0.2-0.8 M with
# 5,000): the size effect is confounded with training length. Here every size (15 50 150 500 1500 5000 slices per
# dataset, the same slices as the original runs for each seed, seeds 0-2) trains the decoder for the SAME number
# of training samples: one run per size and seed to 600k samples, constant learning rate, no early stopping, the
# decoder saved after 6k, 20k, 60k, 200k and 600k samples (arms sam-enc_sam-dec_n<N>_s<B>, 90 decoders). All are
# tested on in-domain, zero-shot and knee as before; analyse_fixed.py crosses size with budget.
#
# BEFORE: run_curve_experiments.sh and run_knee_experiments.sh have finished (their results are the reference).
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_fixed_budget.sh
# Finished runs (DONE for every budget) are skipped on resubmission; only the pool encoding is repeated.
#
# FIRST MINUTES - tail the .output log:
#   "device: AMD Instinct MI355X", "pool <dataset>: N slices" for 11 datasets, "pool: 29382 slices (57.4 GB ...)"
#   "[check] encoder output vs stored features ... cosine min 1.00000"
# AFTER ENCODING (~8 min): "sam-enc_sam-dec_n15 seed 0: 165 slices (... with a structure) from 11 datasets, lr 0.0001,
#   budgets 6,000, 20,000, 60,000, 200,000, 600,000 training samples (= 36.4, ... epochs); validation Dice before
#   training 0.795", then one line per budget: "6,000 samples (36.4 epochs, 188 steps): val Dice 0.8... | 0.1 min
#   training". A 600k run takes ~6-8 min; 18 runs ~2-2.5 h; then the evaluation (~45 min) and the tables.
#SBATCH --job-name=samri-fixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=07:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-fixed-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-fixed-%j.error
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
KNEE=$ROOT/Eval_results/embedding_research_knee
OUT=$ROOT/Eval_results/decoder_swap
RUNS=$OUT/runs
CURVE=$OUT/runs_curve
FIXED=$OUT/runs_fixed
CKPT=$ROOT/pretrained_ckpt
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
BUDGETS="${BUDGETS:-6000 20000 60000 200000 600000}"
for f in "$INDOMAIN/features/sam_vit_b__sam/DONE" "$ZEROSHOT/features/sam_vit_b__sam/DONE" \
         "$KNEE/features/sam_vit_b__sam/DONE" "$RUNS/val_rows.json"; do
    [ -f "$f" ] || { echo "Missing $f - run run_decoder_experiments.sh / run_knee_experiments.sh first"; exit 1; }
done
DEC_LR=$(cat "$OUT/chosen_lr.txt" 2>/dev/null || echo 1e-4)       # the decoder arms' learning rate
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"
echo "decoder learning rate: $DEC_LR; budgets (training samples): $BUDGETS"

# ---- 1. training: every size and seed to the largest budget (one process: the pool is encoded once) ----------
python embedding_assess/train_decoder_curve.py --work-dir "$INDOMAIN" --val-rows "$RUNS/val_rows.json" \
    --ckpt-dir "$CKPT" --out-dir "$FIXED" --workers "$WORKERS" --lr-decoder "$DEC_LR" \
    --sizes 15 50 150 500 1500 5000 --seeds 0 1 2 --modes decoder --fixed-samples $BUDGETS \
    || { echo "!!!!! FAILED: train_decoder_curve (finished runs are kept; resubmit to continue)"; exit 1; }

# ---- 2. evaluation of the 90 decoders on in-domain, zero-shot and knee ------------------------------------------
echo; echo "===== evaluation ====="
python embedding_assess/eval_decoder.py --indomain-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" \
    --extra-test "knee=$KNEE" --runs-dir "$FIXED" --ckpt-dir "$CKPT" --out-dir "$OUT/results_fixed" \
    || { echo "!!!!! FAILED: evaluation"; exit 1; }

# ---- 3. tables: size x budget, with the original early-stopped runs as reference -------------------------------
echo; echo "===== tables ====="
REF=""
[ -f "$OUT/results_knee/eval_instances.csv" ] && REF="--reference $OUT/results_knee/eval_instances.csv --reference-runs $CURVE"
python embedding_assess/analyse_fixed.py --instances "$OUT/results_fixed/eval_instances.csv" --runs "$FIXED" \
    $REF --out-dir "$OUT/results_fixed/tables" || echo "!!!!! FAILED: tables"
echo; echo "all steps finished: $(date)"
