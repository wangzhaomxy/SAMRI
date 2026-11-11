#!/bin/bash --login
# Does training on a SIMILAR dataset help on unseen ones? 1 GPU (MI355X, --qos=gpu), ~1-2 h.
#
#   new knee test set   test splits of MSK_PD and MSK_FLASH (unseen by every 11-dataset model; knee MRI
#                       with the same six targets as MSK_T2, other contrasts), plus MSK_T2's own
#                       test split (seen only by the new arms; a check that they learned it)
#   new arms            decoder trained on the 11 clean datasets + MSK_T2: 150 and 500 slices per dataset,
#                       seeds 0-2 ("sam-enc_sam-dec_n<N>_plus-MSK_T2"). Seed k uses exactly the 11-dataset
#                       slices of the existing run sam-enc_sam-dec_n<N> seed k, so plus - base isolates MSK_T2.
#                       Validation stays on the same 165 slices of the 11 datasets.
#   existing models     every trained model (decoder-only, decoder LoRA, data-size runs, encoder + decoder LoRA)
#                       is also tested on the knee set.
#
# BEFORE submitting: run_decoder_experiments.sh, run_lora_experiments.sh and run_curve_experiments.sh have
# finished (their runs are reused). Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_knee_experiments.sh
# Every step is skipped on resubmission once done.
#
# FIRST MINUTES - tail the .output log:
#   "Sampled ... slices (0 train, ... test, 3 datasets)" - the knee test set
#   "[sanity] sam_vit_b own decoder ... mean Dice" for the knee set (untrained SAM on knee slices)
# LATER (the new arms): "pool MSK_T2 (extra, from ...): N slices", "[check] ... cosine min 1.00000",
#   "sam-enc_sam-dec_n150 seed 0: done - skipping" (the existing runs are reused), then
#   "sam-enc_sam-dec_n150_plus-MSK_T2 seed 0: 1785 slices from 12 datasets ... validation Dice before training 0.795"
#SBATCH --job-name=samri-knee
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-knee-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-knee-%j.error
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
TRAINTEST=$ROOT/Datasets/SAMRI_train_test
INDOMAIN=$ROOT/Eval_results/embedding_research_clean
ZEROSHOT=$ROOT/Eval_results/embedding_research_zeroshot
KNEE=$ROOT/Eval_results/embedding_research_knee
OUT=$ROOT/Eval_results/decoder_swap
RUNS=$OUT/runs
RUNS_LORA=$OUT/runs_lora
CURVE=$OUT/runs_curve
CKPT=$ROOT/pretrained_ckpt
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
for f in "$INDOMAIN/features/sam_vit_b__sam/DONE" "$ZEROSHOT/features/sam_vit_b__sam/DONE" "$RUNS/val_rows.json"; do
    [ -f "$f" ] || { echo "Missing $f - run run_decoder_experiments.sh first"; exit 1; }
done
for ds in MSK_T2 MSK_PD MSK_FLASH; do
    [ -d "$TRAINTEST/$ds/testing" ] || { echo "Missing $TRAINTEST/$ds/testing"; exit 1; }
done
DEC_LR=$(cat "$OUT/chosen_lr.txt" 2>/dev/null || echo 1e-4)
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"
echo "decoder learning rate: $DEC_LR"
for n in 150 500; do
    echo "existing 11-dataset runs, $n per dataset: $(ls -d "$CURVE"/sam-enc_sam-dec_n$n/seed*/DONE 2>/dev/null | wc -l) of 3 seeds"
done

# ---- 1. knee test set: every test slice of MSK_PD, MSK_FLASH, MSK_T2; SAM and MedSAM features -----------
echo; echo "===== knee test set ====="
[ -f "$KNEE/manifest.csv" ] || python embedding_assess/build_manifest.py --zeroshot-root "$TRAINTEST/" \
    --datasets MSK_PD MSK_FLASH MSK_T2 --work-dir "$KNEE" --n-test 0 || { echo "!!!!! FAILED: knee manifest"; exit 1; }
for ep in "sam_vit_b sam" "medsam_vit_b medsam"; do
    set -- $ep
    python embedding_assess/extract_features.py --work-dir "$KNEE" --ckpt-dir "$CKPT" \
        --encoder "$1" --pipeline "$2" --layers neck --workers "$WORKERS" \
        || echo "!!!!! FAILED: knee features $1 / $2"
done

# ---- 2. new arms: the 11 datasets + MSK_T2 (existing 11-dataset runs are skipped, not retrained) ---------
echo; echo "===== 11 datasets + MSK_T2 ====="
python embedding_assess/train_decoder_curve.py --work-dir "$INDOMAIN" --val-rows "$RUNS/val_rows.json" \
    --ckpt-dir "$CKPT" --out-dir "$CURVE" --workers "$WORKERS" --lr-decoder "$DEC_LR" \
    --sizes 150 500 --seeds 0 1 2 --modes decoder --variety-ks \
    --extra-root "$TRAINTEST" --extra-datasets MSK_T2 \
    || echo "!!!!! FAILED: train_decoder_curve (finished runs are kept; resubmit to continue)"

# ---- 3. encoder + decoder LoRA arms: rerun their encoder on the knee images ---------------------------------
echo; echo "===== encoder + decoder LoRA on the knee set ====="
for arm in sam-enc-lora_sam-dec-lora_small sam-enc-lora_sam-dec-lora_n1500 sam-enc-lora_sam-dec-lora_all11; do
    [ -f "$RUNS_LORA/$arm/seed0/DONE" ] || { echo "$arm: no finished run - skipping"; continue; }
    [ -f "$RUNS_LORA/$arm/seed0/eval_instances_knee.csv" ] && { echo "$arm: knee evaluation done - skipping"; continue; }
    python embedding_assess/train_lora.py --arm $arm --seed 0 --work-dir "$INDOMAIN" \
        --val-rows "$RUNS/val_rows.json" --ckpt-dir "$CKPT" --out-dir "$RUNS_LORA" --workers "$WORKERS" \
        --eval-only "knee=$KNEE" || echo "!!!!! FAILED: knee evaluation of $arm"
done

# ---- 4. one evaluation of every model on in-domain, zero-shot and knee ---------------------------------------
echo; echo "===== evaluation ====="
EXTRA_INST=$(ls "$RUNS_LORA"/sam-enc-lora_sam-dec-lora_{small,n1500,all11}/seed*/eval_instances{,_knee}.csv 2>/dev/null)
python embedding_assess/eval_decoder.py --indomain-dir "$INDOMAIN" --zeroshot-dir "$ZEROSHOT" \
    --extra-test "knee=$KNEE" --runs-dir "$RUNS" "$CURVE" --ckpt-dir "$CKPT" \
    --out-dir "$OUT/results_knee" --extra-instances $EXTRA_INST \
    || { echo "!!!!! FAILED: evaluation"; exit 1; }

# ---- 5. tables -----------------------------------------------------------------------------------------------------
python embedding_assess/analyse_knee.py --instances "$OUT/results_knee/eval_instances.csv" \
    --out-dir "$OUT/results_knee/tables" || echo "!!!!! FAILED: knee tables"
echo; echo "all steps finished: $(date)"
