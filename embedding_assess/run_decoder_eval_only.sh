#!/bin/bash --login
# Re-evaluate the swapped-encoder decoders with other SAMRI checkpoints (no training, ~5-10 min, 1 x MI355X).
#
# Why: pretrained_ckpt/samri_vitb_box.pth and samri_vitb_bp.pth did not reproduce the published SAMRI Dice
# on the same 1,473 in-domain structures (0.838 / 0.769 vs 0.887 / 0.891), while SAM ViT-B and MedSAM did.
# This job tests the epoch checkpoints samri_vitb_box_177.pth and samri_vitb_bp_200.pth beside the old two.
#
# It needs what run_decoder_experiments.sh already made: the in-domain and zero-shot features and
# decoder_swap/runs/. Results go to a NEW folder (decoder_swap/results_ckpt_check); results/ is untouched.
#
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_decoder_eval_only.sh
# Other checkpoints:  sbatch --export=ALL,SAMRI_CKPTS="/path/a.pth /path/b.pth" embedding_assess/run_decoder_eval_only.sh
#
# FIRST MINUTES - tail the .output log:
#   the ckpt_check block first - a genuine SAMRI checkpoint shows  image_encoder max |diff| 0  and  prompt_encoder max |diff| 0  vs SAM ViT-B
#   then one "prompt encoder of <file> vs SAM's: ... identical" line per SAMRI checkpoint,
#   then "<system> (seed 0) on in-domain: median Dice ..." lines, about 15 s each.
#SBATCH --job-name=samri-dec-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-dec-eval-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-dec-eval-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
python -c "import torch, numpy, nibabel, scipy.stats" \
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
CKPT=$ROOT/pretrained_ckpt
OUT=$ROOT/Eval_results/decoder_swap
SAMRI_CKPTS=${SAMRI_CKPTS:-"$CKPT/samri_vitb_box_177.pth $CKPT/samri_vitb_bp_200.pth $CKPT/samri_vitb_box.pth $CKPT/samri_vitb_bp.pth"}
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"
for p in $SAMRI_CKPTS; do [ -f "$p" ] || echo "WARNING: not found: $p"; done

echo; echo "===== checkpoint check (vs SAM ViT-B) ====="
python embedding_assess/ckpt_check.py --sam "$CKPT/sam_vit_b_01ec64.pth" --ckpts $SAMRI_CKPTS \
    || echo "!!!!! FAILED: ckpt_check"

echo; echo "===== evaluation ====="
python embedding_assess/eval_decoder.py \
    --indomain-dir "$ROOT/Eval_results/embedding_research_clean" \
    --zeroshot-dir "$ROOT/Eval_results/embedding_research_zeroshot" \
    --runs-dir "$OUT/runs" --ckpt-dir "$CKPT" --samri-ckpts $SAMRI_CKPTS \
    --out-dir "$OUT/results_ckpt_check" \
    || echo "!!!!! FAILED: evaluation"
echo; echo "finished: $(date)"
