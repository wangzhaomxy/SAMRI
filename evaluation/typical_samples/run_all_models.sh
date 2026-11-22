#!/bin/bash --login
# All-model inference on the candidate slices of the qualitative figure, in ONE interactive session with ONE
# NVIDIA GPU (H100); then the typical slice of every structure type is chosen from the Dice of this run.
#
# RUN (inside the H100 session):
#   bash -l /scratch/user/s4670484/Eval_results/typical_samples/all_models/run_all_models.sh
# Quick test first (3 slices per model, a few minutes, separate folder):
#   LIMIT=3 TAG=test bash -l /scratch/user/s4670484/Eval_results/typical_samples/all_models/run_all_models.sh
#
# Models, one after another on the GPU, each in its own conda environment (/scratch/user/s4670484/conda/<env>):
#   sam_med2d  -> SAM ViT-B, SAMRI box, SAMRI box+point, MedSAM, SAM-Med2D
#                 (scikit-image 0.25.0 and matplotlib 3.10.9 were added to it on 23 Nov 2026 for MedSAM and
#                  the pictures; numpy, scipy, pillow, torch and torchvision unchanged)
#   mcp-medsam -> MCP-MedSAM (its CLIP models come from the Hugging Face cache, offline)
#   samed      -> SAMed
#   Medical-SAM-Adapter is not run.
#   Then collect_all_models.py (sam_med2d, CPU): checks, chosen.csv, npz/, png/.
# Output: /scratch/user/s4670484/Eval_results/typical_samples/all_models/<TAG, default run_h100>/
#   run.log, dice_<model>.csv, masks_<model>.npz, all_models_dice.csv, chosen.csv, npz/, png/
# Results contain images of non-public datasets: never commit them.

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=/scratch/user/s4670484/SAMRI
ENVS=/scratch/user/s4670484/conda
LIST=$REPO/evaluation/typical_samples/typical_samples_all_models.csv
OUT=$HERE/${TAG:-run_h100}
LIM=${LIMIT:+--limit $LIMIT}
mkdir -p "$OUT" || exit 1
exec > >(tee -a "$OUT/run.log") 2>&1
echo "start: $(date)  host: $(hostname)  out: $OUT${LIMIT:+  (first $LIMIT slices only)}"

for f in "$LIST" "$HERE/run_candidates.py" "$HERE/collect_all_models.py" "$REPO/evaluation/typical_samples/typical_samples.py"; do
    [ -f "$f" ] || { echo "missing: $f"; exit 1; }
done
nvidia-smi -L || { echo "no NVIDIA GPU visible - run this inside the H100 session"; exit 1; }
module load anaconda3/2023.09-0 && source "$EBROOTANACONDA3/etc/profile.d/conda.sh" \
    || { echo "could not set up conda (module load anaconda3/2023.09-0)"; exit 1; }
use_env () {                     # use_env <env folder name under $ENVS>
    while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
    conda activate "$ENVS/$1"
}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
FAILED=""

for pair in sam_med2d:sam_box sam_med2d:samri_box sam_med2d:samri_bp sam_med2d:medsam sam_med2d:sam_med2d \
            mcp-medsam:mcp_medsam samed:samed; do
    env=${pair%%:*}; model=${pair##*:}
    echo; echo "===== $model   (conda env $ENVS/$env) ====="
    if use_env "$env" && python -u "$HERE/run_candidates.py" "$model" --list "$LIST" --out "$OUT" --device cuda $LIM; then
        :
    else
        echo "!!!!! $model FAILED - the other models go on"; FAILED="$FAILED $model"
    fi
done

echo; echo "===== collect ====="
use_env sam_med2d && python -u "$HERE/collect_all_models.py" --list "$LIST" --out "$OUT" || FAILED="$FAILED collect"

echo; echo "finished: $(date)${FAILED:+   !!!!! FAILED:$FAILED}"
echo "results: $OUT"
