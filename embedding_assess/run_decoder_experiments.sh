#!/bin/bash --login
# Swapped-encoder decoder experiments, 1 GPU (MI300X, --qos=gpu).
#
# Q1  does the encoder matter for SAMRI's decoder?  Identical SAM mask decoders trained
#     with SAMRI's recipe on frozen SAM ViT-B output vs frozen MedSAM-encoder output,
#     on the same 11 clean datasets (150 training slices each), 3 seeds per arm.
# Q2  does more training data improve zero-shot?  small data -> all training slices of
#     the same 11 datasets -> released SAMRI (36 datasets), tested on the zero-shot sets.
#
# BEFORE submitting (once): the in-domain features must exist (run_embedding_experiments.sh
# on embedding_research_clean), and the log folder must exist:
#   mkdir -p /scratch/user/s4670484/Eval_results/decoder_swap/logs
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_decoder_experiments.sh
# Optional: skip the learning-rate pilot with  sbatch --export=ALL,LR=1e-4 ...
#
# FIRST MINUTES - tail the .output log and check:
#   "device: AMD Instinct MI300X"                        GPU visible, right env
#   "Sampled ... slices (0 train, ... test, 6 datasets)" zero-shot manifest built
#   "[sanity] sam_vit_b own decoder ... mean Dice ~0.7-0.9" and the same for medsam_vit_b
#   then the pilot: "[sanity] decoder before training, validation Dice ~0.8"
#   and "epoch   1: ... val Dice" within a few minutes.
# LATER (all-slices arm, step 4): "[check] encoder output now vs stored features ... cosine min ~1.0000"
#   and the per-dataset slice counts / whether a cap was applied.
# A Dice near 0 anywhere means scancel and check.
# Every step writes DONE and is skipped on resubmission, so a cut job loses nothing.
#SBATCH --job-name=samri-dec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=11:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-dec-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-dec-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
# deactivate anything inherited from the submitting shell, then activate (see run_embedding_experiments.sh)
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate samri-mi300
python -c "import torch, torchvision, numpy, nibabel, skimage, scipy.stats" \
    || { echo "samri-mi300 is missing a package (python: $(which python)) - see the .error log; stopping"; exit 1; }
set -uo pipefail                    # no -e: one failed run must not stop the rest
cd "${SLURM_SUBMIT_DIR:-/scratch/user/s4670484/SAMRI}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

ROOT=/scratch/user/s4670484
INDOMAIN=$ROOT/Eval_results/embedding_research_clean     # 11 clean datasets, features already extracted
ZEROSHOT=$ROOT/Eval_results/embedding_research_zeroshot  # built below
OUT=$ROOT/Eval_results/decoder_swap
RUNS=$OUT/runs
CKPT=$ROOT/pretrained_ckpt
MAX_GPU_GB=${MAX_GPU_GB:-100}       # all-slices arm: encoder output kept in GPU memory (2 MiB/slice, MI300X has 192 GB)
ZS_ROOT=$ROOT/Datasets/Zeroshot
SAMRI_CKPTS="$CKPT/samri_vitb_box.pth $CKPT/samri_vitb_bp.pth"   # box / box+point checkpoints (tested with box)
SEEDS="0 1 2"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
mkdir -p "$RUNS"

for c in sam_vit_b__sam medsam_vit_b__medsam; do
    [ -f "$INDOMAIN/features/$c/DONE" ] || { echo "Missing $INDOMAIN/features/$c - run the embedding job first"; exit 1; }
done
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"

# ---- 1. zero-shot test set: every test slice of the zero-shot datasets ----------------
echo; echo "===== zero-shot test set ====="
[ -f "$ZEROSHOT/manifest.csv" ] || python embedding_assess/build_manifest.py \
    --zeroshot-root "$ZS_ROOT/" --work-dir "$ZEROSHOT" --n-test 0 || echo "!!!!! FAILED: zero-shot manifest"
for ep in "sam_vit_b sam" "medsam_vit_b medsam"; do
    set -- $ep
    python embedding_assess/extract_features.py --work-dir "$ZEROSHOT" --ckpt-dir "$CKPT" \
        --encoder "$1" --pipeline "$2" --layers neck --workers "$WORKERS" \
        || echo "!!!!! FAILED: zero-shot features $1 / $2"
done

train () {  # train <arm> <seed> [extra args]
    echo; echo "===== train $1 seed $2 ${*:3} ====="
    python embedding_assess/train_decoder.py --arm "$1" --seed "$2" --work-dir "$INDOMAIN" \
        --ckpt-dir "$CKPT" --out-dir "$RUNS" --workers "$WORKERS" "${@:3}" \
        || echo "!!!!! FAILED: train $1 seed $2 ${*:3}"
}

# ---- 2. learning-rate pilot (SAM arm, seed 0), unless LR was given ----------------------
if [ -z "${LR:-}" ]; then
    if [ -f "$OUT/chosen_lr.txt" ]; then
        LR=$(cat "$OUT/chosen_lr.txt")
    else
        for lr in 1e-5 1e-4; do train sam-enc_sam-dec_small 0 --lr $lr --epochs 60 --tag "_pilot-lr$lr"; done
        LR=$(python - "$RUNS" <<'EOF'
import json, sys, os
best = max(("1e-5", "1e-4"), key=lambda lr: json.load(open(os.path.join(
    sys.argv[1], f"sam-enc_sam-dec_small_pilot-lr{lr}", "seed0", "meta.json")))["best_val_dice"])
print(best)
EOF
)
        echo "$LR" > "$OUT/chosen_lr.txt"
    fi
fi
echo; echo "learning rate for every arm: $LR"

# ---- 3. Q1 arms: small data, 3 seeds each ---------------------------------------------
for seed in $SEEDS; do
    for arm in sam-enc_sam-dec_small medsam-enc_sam-dec_small medsam-enc_medsam-dec_small; do
        train "$arm" "$seed" --lr "$LR"
    done
done

# ---- 4. Q2 middle step: all training slices of the 11 datasets, 1 seed. Encodes the images once at
#      the start and keeps the output in GPU memory (no disk); capped per dataset above MAX_GPU_GB.
train sam-enc_sam-dec_all11 0 --lr "$LR" --max-gpu-gb "$MAX_GPU_GB" --epochs 15 --patience 4

# ---- 5. evaluation and summary ---------------------------------------------------------------
echo; echo "===== evaluation ====="
python embedding_assess/eval_decoder.py --indomain-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" \
    --runs-dir "$RUNS" --ckpt-dir "$CKPT" --samri-ckpts $SAMRI_CKPTS --out-dir "$OUT/results" \
    || echo "!!!!! FAILED: evaluation"
echo; echo "all steps finished: $(date)"
