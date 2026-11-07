#!/bin/bash --login
# One evaluation of every arm (swapped-encoder, decoder LoRA, encoder + decoder LoRA, data-size and data-variety
# runs) and the tables: compute vs accuracy, data size, data variety. 1 GPU (MI355X, --qos=gpu), ~30 min.
#
# Optional (off by default): one more encoder + decoder LoRA point at 1,500 slices per dataset - the same
# slices as the decoder-only data-size run sam-enc_sam-dec_n1500 (seed 0) - to see whether the LoRA advantage
# shrinks with data. Speed not measured; check the LoRA job's "~N min per epoch" first. Turn it on with
#   sbatch --time=12:00:00 --export=ALL,LORA_N1500=1 embedding_assess/run_final_tables.sh
#
# BEFORE submitting: run_lora_experiments.sh and run_curve_experiments.sh have finished.
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_final_tables.sh
#
# FIRST MINUTES - tail the .output log:
#   "found: ..." lines for each run folder; "<system> (seed k) on in-domain: median Dice ..." lines, ~15 s each;
#   "added N structures from .../eval_instances.csv" for each image-trained arm.
#SBATCH --job-name=samri-final
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-final-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-final-%j.error
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
RUNS_LORA=$OUT/runs_lora
CURVE=$OUT/runs_curve
CKPT=$ROOT/pretrained_ckpt
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"
for d in "$RUNS" "$RUNS_LORA" "$CURVE"; do
    echo "found: $d - $(ls -d "$d"/*/seed*/DONE 2>/dev/null | wc -l) finished runs"
done

# ---- optional: encoder + decoder LoRA, 1,500 slices per dataset ------------------------------------------
if [ "${LORA_N1500:-0}" = 1 ]; then
    ENC_LR=$(python -c "import json; print(json.load(open('$RUNS_LORA/sam-enc-lora_sam-dec-lora_small/seed0/meta.json'))['lr'])" \
        2>/dev/null) || { echo "the LoRA job has not finished its encoder + decoder LoRA step - stopping"; exit 1; }
    echo; echo "===== encoder + decoder LoRA, 1,500 slices per dataset (lr $ENC_LR) ====="
    python embedding_assess/train_lora.py --arm sam-enc-lora_sam-dec-lora_n1500 --seed 0 --lr "$ENC_LR" \
        --work-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" --val-rows "$RUNS/val_rows.json" \
        --ckpt-dir "$CKPT" --out-dir "$RUNS_LORA" --workers "$WORKERS" \
        --samples-per-epoch 8192 --epochs 20 --patience 5 \
        || echo "!!!!! FAILED: encoder + decoder LoRA n1500"
fi

# ---- one evaluation of everything --------------------------------------------------------------------------
echo; echo "===== combined evaluation ====="
RUN_ROOTS="$RUNS"; [ -d "$CURVE" ] && RUN_ROOTS="$RUNS $CURVE"
EXTRA_INST=$(ls "$RUNS_LORA"/sam-enc-lora_sam-dec-lora_{small,n1500,all11}/seed*/eval_instances.csv 2>/dev/null)
python embedding_assess/eval_decoder.py --indomain-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" \
    --runs-dir $RUN_ROOTS --ckpt-dir "$CKPT" --out-dir "$OUT/results_all" --extra-instances $EXTRA_INST \
    || { echo "!!!!! FAILED: evaluation"; exit 1; }

# ---- tables ---------------------------------------------------------------------------------------------
POOL=""; [ -f "$CURVE/pool.json" ] && POOL="--pool-json $CURVE/pool.json"
python embedding_assess/analyse_curve.py --instances "$OUT/results_all/eval_instances.csv" \
    --runs $RUN_ROOTS "$RUNS_LORA" $POOL --out-dir "$OUT/results_all/tables" \
    || echo "!!!!! FAILED: tables"
echo; echo "all steps finished: $(date)"
