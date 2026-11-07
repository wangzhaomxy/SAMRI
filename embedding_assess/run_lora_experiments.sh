#!/bin/bash --login
# LoRA arms for the swapped-encoder comparison, 1 GPU (MI355X, --qos=gpu).
#
#   decoder LoRA only (encoder frozen)  sam-enc_sam-dec-lora_{small,all11}       train_decoder.py, stored features
#   encoder + decoder LoRA              sam-enc-lora_sam-dec-lora_{small,all11}  train_lora.py, images
# LoRA rank 4 (alpha 4) on the query and value projections; all original weights frozen. Same data, recipe,
# validation slices and test sets as the decoder arms of run_decoder_experiments.sh; seed 0.
#
# BEFORE submitting: run_decoder_experiments.sh must have finished (it made the in-domain and zero-shot
# features and decoder_swap/runs/val_rows.json); the log folder already exists from that job.
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_lora_experiments.sh
# Every run writes DONE (train_lora.py also last.pt every epoch) and is skipped or resumed on resubmission.
#
# FIRST MINUTES - tail the .output log:
#   "device: AMD Instinct MI355X"
#   decoder-LoRA trial: "decoder LoRA rank 4 ... 24 k trainable parameters",
#     "[sanity] decoder before training, validation Dice 0.795" (LoRA starts as SAM), epochs of ~2 s
# LATER (step 3, encoder + decoder LoRA):
#   "[check] encoder output vs stored features ... cosine min ~1.0000"
#   "[sanity] before training, validation Dice ~0.795"
#   "epoch 1 batch 10/...: ... ~N min per epoch" - the speed estimate for everything that follows
# A Dice near 0 or a cosine well below 0.999 means scancel and check.
#SBATCH --job-name=samri-lora
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=20:00:00             # encoder LoRA speed is not measured yet; resumable, so a cut costs a resubmission
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-lora-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-lora-%j.error
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
RUNS=$OUT/runs            # decoder-LoRA arms live beside the decoder arms (eval_decoder.py loads them there)
RUNS_LORA=$OUT/runs_lora  # encoder + decoder LoRA arms (evaluate themselves)
CKPT=$ROOT/pretrained_ckpt
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
for f in "$INDOMAIN/features/sam_vit_b__sam/DONE" "$ZEROSHOT/features/sam_vit_b__sam/DONE" "$RUNS/val_rows.json"; do
    [ -f "$f" ] || { echo "Missing $f - run run_decoder_experiments.sh first"; exit 1; }
done
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"

best_lr () {  # best_lr <runs dir> <arm> <lr>... -> the lr whose trial run has the best validation Dice
    python - "$@" <<'EOF'
import json, os, sys
d, arm, lrs = sys.argv[1], sys.argv[2], sys.argv[3:]
val = {}
for lr in lrs:
    p = os.path.join(d, f"{arm}_lr{lr}", "seed0", "meta.json")
    if os.path.isfile(p):
        val[lr] = json.load(open(p))["best_val_dice"]
print(max(val, key=val.get) if val else "")
EOF
}
promote () {  # promote <runs dir> <arm> <lr>: the winning trial run IS the seed-0 run (same settings) -> copy it
    local src="$1/$2_lr$3/seed0" dst="$1/$2/seed0"
    [ -f "$dst/DONE" ] && return
    mkdir -p "$1/$2" && cp -r "$src" "$dst"
    if [ -f "$dst/eval_instances.csv" ]; then  # system id without the trial tag
        python - "$dst/eval_instances.csv" "$2" <<'EOF'
import csv, sys
p, arm = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(p)))
for r in rows:
    r["system"], r["name"] = arm, r["name"].rsplit(" (lr", 1)[0]
w = csv.DictWriter(open(p, "w", newline=""), fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
EOF
    fi
    echo "promoted $src -> $dst"
}

# ---- 1. decoder LoRA, small data: learning-rate trial (each run = a full seed-0 run) -------------------
echo; echo "===== decoder LoRA, small data ====="
DEC_LRS="1e-4 5e-4 1e-3"
for lr in $DEC_LRS; do
    python embedding_assess/train_decoder.py --arm sam-enc_sam-dec-lora_small --seed 0 --lr $lr --tag "_lr$lr" \
        --work-dir "$INDOMAIN" --ckpt-dir "$CKPT" --out-dir "$RUNS" --workers "$WORKERS" \
        || echo "!!!!! FAILED: decoder LoRA small lr $lr"
done
DEC_LR=$(best_lr "$RUNS" sam-enc_sam-dec-lora_small $DEC_LRS)
[ -n "$DEC_LR" ] || { echo "!!!!! FAILED: no decoder-LoRA trial finished - stopping"; exit 1; }
echo "decoder LoRA learning rate: $DEC_LR"
promote "$RUNS" sam-enc_sam-dec-lora_small "$DEC_LR"

# ---- 2. decoder LoRA, all training slices (encodes the images once, as the decoder all-slices arm) ------
echo; echo "===== decoder LoRA, all training slices ====="
python embedding_assess/train_decoder.py --arm sam-enc_sam-dec-lora_all11 --seed 0 --lr "$DEC_LR" \
    --work-dir "$INDOMAIN" --ckpt-dir "$CKPT" --out-dir "$RUNS" --workers "$WORKERS" \
    --max-gpu-gb 100 --epochs 15 --patience 4 \
    || echo "!!!!! FAILED: decoder LoRA all slices"

# ---- 3. encoder + decoder LoRA, small data: learning-rate trial ---------------------------------------
echo; echo "===== encoder + decoder LoRA, small data ====="
ENC_LRS="1e-4 5e-4"
for lr in $ENC_LRS; do
    python embedding_assess/train_lora.py --arm sam-enc-lora_sam-dec-lora_small --seed 0 --lr $lr --tag "_lr$lr" \
        --work-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" --val-rows "$RUNS/val_rows.json" \
        --ckpt-dir "$CKPT" --out-dir "$RUNS_LORA" --workers "$WORKERS" --epochs 60 --patience 10 \
        || echo "!!!!! FAILED: encoder + decoder LoRA small lr $lr"
done
ENC_LR=$(best_lr "$RUNS_LORA" sam-enc-lora_sam-dec-lora_small $ENC_LRS)
[ -n "$ENC_LR" ] || { echo "!!!!! FAILED: no encoder + decoder LoRA trial finished - stopping"; exit 1; }
echo "encoder + decoder LoRA learning rate: $ENC_LR"
promote "$RUNS_LORA" sam-enc-lora_sam-dec-lora_small "$ENC_LR"

# ---- 4. encoder + decoder LoRA, all training slices (same 51,198 slices; 8,192 per epoch) --------------
echo; echo "===== encoder + decoder LoRA, all training slices ====="
python embedding_assess/train_lora.py --arm sam-enc-lora_sam-dec-lora_all11 --seed 0 --lr "$ENC_LR" \
    --work-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" --val-rows "$RUNS/val_rows.json" \
    --ckpt-dir "$CKPT" --out-dir "$RUNS_LORA" --workers "$WORKERS" \
    --samples-per-epoch 8192 --epochs 20 --patience 5 \
    || echo "!!!!! FAILED: encoder + decoder LoRA all slices"

# ---- 5. evaluation of every arm together -----------------------------------------------------------------
echo; echo "===== evaluation ====="
EXTRA=$(ls "$RUNS_LORA"/sam-enc-lora_sam-dec-lora_small/seed*/eval_instances.csv \
           "$RUNS_LORA"/sam-enc-lora_sam-dec-lora_all11/seed*/eval_instances.csv 2>/dev/null)
python embedding_assess/eval_decoder.py --indomain-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" \
    --runs-dir "$RUNS" --ckpt-dir "$CKPT" --out-dir "$OUT/results_lora" --extra-instances $EXTRA \
    || echo "!!!!! FAILED: evaluation"
echo; echo "all steps finished: $(date)"
