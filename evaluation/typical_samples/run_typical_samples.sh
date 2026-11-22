#!/bin/bash --login
# Typical test slices for the qualitative figure of the paper (replaces the hand-picked slices, one of which
# is a CT slice). Runs SAM ViT-B, SAMRI box, SAMRI box+point and MedSAM on the 196 structures of
# typical_samples.csv (33 structure types, up to 6 candidates each, chosen by select_typical_samples.py),
# checks every Dice against the benchmark records, and saves the masks and preview pictures. Nothing is
# trained. 1 MI355X, about 5-10 minutes.
#
# WHERE THE FILES GO: this folder is evaluation/typical_samples/ of the SAMRI repository. Commit and push it
# from the Mac, then update the repo on Bunya:  cd /scratch/user/s4670484/SAMRI && git pull --ff-only
# BEFORE submitting (once):
#   mkdir -p /scratch/user/s4670484/Eval_results/typical_samples/logs
# SUBMIT from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch evaluation/typical_samples/run_typical_samples.sh
#
# The SAMRI checkpoints are the two lines SAMRI_BOX / SAMRI_BP below. Other files, without editing:
#   sbatch --export=ALL,SAMRI_BOX=/path/box.pth,SAMRI_BP=/path/bp.pth,TAG=try2 evaluation/typical_samples/run_typical_samples.sh
#
# FIRST MINUTES - in the .output log, in this order:
#   "device: AMD Instinct MI355X"
#   "196 structures listed in typical_samples.csv; all image and mask files found and read"
#   "===== SAM ViT-B (box) =====", a table, then
#   "SAM ViT-B (box): 196 of 196 structures within 0.005 of the records ...  ->  REPRODUCES the records"
#        = the data and the code are those of the benchmark. If it says DOES NOT, stop (scancel) and check.
#   "SAMRI box (box): ...  ->  REPRODUCES the records"
#        = this checkpoint is the one behind the paper's tables.
#        "DOES NOT reproduce" = it is another checkpoint: do not use the pictures of this job.
#   then SAMRI box+point ("agrees with the records": its point is random, so small differences are normal)
#   and MedSAM ("REPRODUCES").
# At the end: "===== summary =====" with one line per model.
#
# ONLY IF a SAMRI checkpoint does not reproduce the records: rank a folder of checkpoints against the records
# (about 7 s per checkpoint; set the pattern to where the epoch checkpoints are):
#   sbatch --time=01:30:00 --export=ALL,SCAN="/scratch/user/s4670484/Model_save/*.pth",SCAN_PROMPT=box evaluation/typical_samples/run_typical_samples.sh
#   (SCAN_PROMPT=box for the box model, SCAN_PROMPT=box+point for the box+point model)
# The log then lists every checkpoint with its mean distance from the records; the right one is near 0.
#
# AFTERWARDS copy the result folder to the Mac, into the SAMRI paper folder (results do not go into git):
#   scp -r s4670484@bunya.rcc.uq.edu.au:/scratch/user/s4670484/Eval_results/typical_samples/v2 <SAMRI paper folder>/Experiment_results/typical_samples_v2
#SBATCH --job-name=samri-typical
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/typical_samples/logs/slurm-typical-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/typical_samples/logs/slurm-typical-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
python -c "import torch, numpy, nibabel, skimage, matplotlib" \
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
HERE=evaluation/typical_samples
SAM_CKPT=$ROOT/pretrained_ckpt/sam_vit_b_01ec64.pth
MEDSAM_CKPT=$ROOT/pretrained_ckpt/medsam_vit_b.pth
SAMRI_BOX=${SAMRI_BOX:-$ROOT/pretrained_ckpt/samri_vitb_box.pth}     # the box model of the paper's tables
SAMRI_BP=${SAMRI_BP:-$ROOT/pretrained_ckpt/samri_vitb_bp.pth}        # the box+point model of the paper's tables
TAG=${TAG:-v2}
OUT=$ROOT/Eval_results/typical_samples/$TAG
SCAN=${SCAN:-}
SCAN_PROMPT=${SCAN_PROMPT:-box}

[ -f "$HERE/typical_samples.py" ] && [ -f "$HERE/typical_samples.csv" ] \
    || { echo "Not in the repo root, or $HERE is missing: $(pwd)"; exit 1; }
for f in "$SAM_CKPT" "$MEDSAM_CKPT" "$SAMRI_BOX" "$SAMRI_BP"; do
    [ -f "$f" ] || { echo "Missing checkpoint: $f"; exit 1; }
done
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.hip); \
assert torch.cuda.is_available(), 'no GPU visible'; print('device:', torch.cuda.get_device_name(0))" || exit 1
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)"
echo "SAMRI box:       $SAMRI_BOX"
echo "SAMRI box+point: $SAMRI_BP"
mkdir -p "$OUT"

ARGS=(--samples "$HERE/typical_samples.csv" --data-root "$ROOT/Datasets" --repo-root "$(pwd)"
      --sam-ckpt "$SAM_CKPT" --medsam-ckpt "$MEDSAM_CKPT" --samri-box-ckpt "$SAMRI_BOX" --samri-bp-ckpt "$SAMRI_BP"
      --out-dir "$OUT" --device cuda)
if [ -n "$SCAN" ]; then
    python -u "$HERE/typical_samples.py" "${ARGS[@]}" --scan-ckpts "$SCAN" --scan-prompt "$SCAN_PROMPT"
else
    python -u "$HERE/typical_samples.py" "${ARGS[@]}"
fi
RC=$?
echo
case $RC in
    0) echo "finished: $(date)  ->  $OUT" ;;
    2) echo "finished: $(date), but at least one model DOES NOT MATCH the records (see the summary above)  ->  $OUT" ;;
    *) echo "!!!!! FAILED with exit code $RC: $(date)" ;;
esac
exit $RC
