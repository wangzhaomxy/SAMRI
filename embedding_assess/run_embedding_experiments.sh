#!/bin/bash --login
# Embedding experiments (CKA + probes) - feature extraction and analysis, 1 GPU.
#
# BEFORE submitting (once, on a CPU node):
#   python embedding_assess/build_manifest.py \
#       --trained-root /scratch/user/s4670484/Datasets/SAMRI_train_test/ \
#       --zeroshot-root /scratch/user/s4670484/Datasets/Zeroshot/ \
#       --work-dir /scratch/user/s4670484/Eval_results/embedding_research/
#   mkdir -p /scratch/user/s4670484/Eval_results/embedding_research/logs
# Submit from the repo root:
#   sbatch embedding_assess/run_embedding_experiments.sh
#
# FIRST MINUTES: tail the .output log. Within the first condition you should see
#   "device: AMD Instinct MI300X ..."                     (GPU visible, right env)
#   "[sanity] sam_vit_b own decoder ... mean Dice ~0.7-0.9" (preprocessing + ckpt OK)
#   "[check] ... cosine 1.0000"                            (matches SAMRI's stored embeddings)
# and later "[sanity] medsam_vit_b ... medsam pipeline: mean Dice ~0.7-0.9".
# A Dice near 0 or a cosine well below 0.999 means stop the job (scancel) and check.
#
# Card: 1 x MI300X under --qos=gpu with samri-mi300 (bunya_hpc_settings.md).
# For H100 instead: --partition=gpu_cuda, conda env samri, and the H100 --gres name
# (not in bunya_hpc_settings.md - check with `sinfo -p gpu_cuda -o "%G"`).
# Each condition writes features/<cond>/DONE and is skipped on resubmission.
#SBATCH --job-name=samri-emb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi300x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/embedding_research/logs/slurm-emb-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/embedding_research/logs/slurm-emb-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
# sbatch copies the submitting shell's environment into the job. If an env was
# active there, `conda activate` only re-activates it in place, behind the
# anaconda bin that `module load` just put first in PATH -> base python, no torch.
# Deactivate everything inherited first.
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate samri-mi300
python -c "import torch, torchvision" \
    || { echo "samri-mi300 not usable (python: $(which python)) - stopping"; exit 1; }
set -uo pipefail                    # after conda (its scripts are not -u safe); no -e: one failed condition must not stop the rest
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

ROOT=/scratch/user/s4670484
# override: sbatch --export=ALL,WORK=/path -o /path/logs/slurm-emb-%j.output -e /path/logs/slurm-emb-%j.error ...
# (#SBATCH lines cannot use variables, so the log paths above do not follow WORK on their own)
WORK=${WORK:-$ROOT/Eval_results/embedding_research}
CKPT=$ROOT/pretrained_ckpt
STORED=$ROOT/Datasets/Embedding_train
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))

[ -f "$WORK/manifest.csv" ] || { echo "Missing $WORK/manifest.csv - run build_manifest.py first"; exit 1; }
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"

extract () {  # extract <encoder> <pipeline> [extra args]
    echo; echo "===== $1 / $2 ${*:3} ====="
    python embedding_assess/extract_features.py --work-dir "$WORK" --ckpt-dir "$CKPT" \
        --encoder "$1" --pipeline "$2" --workers "$WORKERS" "${@:3}" \
        || echo "!!!!! FAILED: $1 / $2 ${*:3}"
}

# 2x2 design: each encoder under both pipelines (SAM pad / MedSAM stretch)
extract sam_vit_b    sam    --stored-embedding-root "$STORED"
extract medsam_vit_b medsam
extract sam_vit_b    medsam
extract medsam_vit_b sam
# references: untrained ViT-B (not a floor, see README), small-noise ceiling (CKA only), ViT-H (claim 1)
extract random_vit_b sam
extract random_vit_b medsam
extract sam_vit_b    sam    --noise-std 1 --rows test
extract sam_vit_b    medsam --noise-std 1 --rows test
extract sam_vit_h    sam    --layers neck

echo; echo "===== CKA ====="
python embedding_assess/cka.py --work-dir "$WORK" || echo "!!!!! FAILED: cka"
echo; echo "===== probes ====="
python embedding_assess/probe.py --work-dir "$WORK" || echo "!!!!! FAILED: probes"
echo; echo "all steps finished: $(date)"
