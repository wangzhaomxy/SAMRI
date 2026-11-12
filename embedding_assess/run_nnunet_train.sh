#!/bin/bash --login
# nnU-Net comparison, step 2: one 2D nnU-Net per dataset (array task i = datasets.json index i), 1 MI355X each,
# at most 3 at a time. Each task trains (nnU-Net defaults: 1000 epochs, fold 0 = the dataset's patient-wise
# training/validation split), runs nnU-Net's validation, predicts its test slices and writes
# Eval_results/nnunet_knee/eval/nnUNetTrainer/eval_instances_<dataset>.csv. The MSK_T2 task also tests MSK_PD
# and MSK_FLASH (same six classes); no zero-shot set is tested.
#
# BEFORE: run_nnunet_prepare.sh has finished. Submit from the repo root.
# 1) SMOKE TEST first (~30 min): smallest dataset, 5 epochs, the whole chain; prints the time per epoch:
#      sbatch --array=0 --time=01:00:00 --export=ALL,SMOKE=1 embedding_assess/run_nnunet_train.sh
#    Check: "device: AMD Instinct MI355X", "Epoch 0" ... "Epoch 4" with falling train_loss, "epoch time: median
#    X s ... -> 1000 epochs = Y h", "nnU-Net's own validation ...", "prediction: 23 test slices", a median Dice
#    line, no Traceback.
# 2) FULL RUN (12 tasks): if Y h + ~1 h is above --time below, raise --time on the command line:
#      sbatch embedding_assess/run_nnunet_train.sh
#    Resumable: resubmitting continues unfinished tasks from their last 50-epoch checkpoint
#    (e.g. sbatch --array=3,7 embedding_assess/run_nnunet_train.sh for the tasks that ran out of time).
# 3) LESION FIX (after the second run_nnunet_prepare.sh): the one-label lesion datasets, array indices 12-14:
#      sbatch --array=12-14 embedding_assess/run_nnunet_train.sh
#    Check: "scoring: one lesion class, per lesion", "first-run result ... -> ....csv.superseded", "per lesion:
#    label 2 median ..., higher labels median ..." (higher labels no longer all missed), "whole slice ...".
#SBATCH --job-name=samri-nnunet
#SBATCH --array=0-11%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-nnunet-%A_%a.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-nnunet-%A_%a.error
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
source /scratch/user/s4670484/conda/nnunet-mi355x-venv/bin/activate
python -c "import torch, nnunetv2; assert torch.cuda.is_available(), 'no GPU visible'" \
    || { echo "nnU-Net venv or GPU missing (python: $(which python)); stopping"; exit 1; }
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/user/s4670484/SAMRI}"

NN=/scratch/user/s4670484/Eval_results/nnunet_knee
[ -f "$NN/datasets.json" ] || { echo "Missing $NN/datasets.json - run run_nnunet_prepare.sh first"; exit 1; }
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export nnUNet_compile=f nnUNet_n_proc_DA=$WORKERS nnUNet_def_n_proc=$WORKERS
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"

if [ "${SMOKE:-0}" = "1" ]; then
    echo "SMOKE TEST: QIN-PROSTATE-Lesion, 5 epochs"
    python embedding_assess/nnunet_run.py --nnunet-root "$NN" --dataset QIN-PROSTATE-Lesion \
        --trainer nnUNetTrainer_5epochs --workers "$WORKERS" || { echo "!!!!! FAILED: smoke test"; exit 1; }
else
    python embedding_assess/nnunet_run.py --nnunet-root "$NN" --index "${SLURM_ARRAY_TASK_ID:-0}" \
        --workers "$WORKERS" || { echo "!!!!! FAILED: nnU-Net task ${SLURM_ARRAY_TASK_ID:-0}"; exit 1; }
fi
echo "finished: $(date)"
