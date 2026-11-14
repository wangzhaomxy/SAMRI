#!/bin/bash --login
# SAM ViT-B (original weights, no training) on EVERY test set with the paper's test pipeline
# (evaluation/test_vis.py -> utils/visual.get_test_record_from_ds), which scores three prompts per
# structure: box, point, and box + point. The box + point numbers are what this job is for
# (SAM ViT-B with the same box + point prompt as SAMRI box+point).
#
#   test sets   every dataset folder with a testing/ split under Datasets/SAMRI_train_test and Datasets/Zeroshot
#   prompts     tight ground-truth box (no jitter); one foreground point from the middle 45-55% of the
#               structure's pixels (the paper's rule). The point is drawn at random, so each dataset is
#               seeded with 0: a rerun gives the same points. (The published SAMRI box+point runs were
#               not seeded, so their points are not the same ones.)
#   resumable   one dataset at a time; a finished dataset is skipped on resubmission.
#
# BEFORE submitting (once):
#   mkdir -p /scratch/user/s4670484/Eval_results/sam_bp_all/logs
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch evaluation/run_sam_bp_all.sh
#
# FIRST MINUTES - in the .output log:
#   "device: AMD Instinct MI355X"
#   "SAMRI_train_test: N datasets with a testing split" and "Zeroshot: 6 datasets ..."
#   "===== SAMRI_train_test / <first dataset> =====", "Testing datasets: ['<first dataset>']",
#   then (1-10 min later) "Elapsed time: ... seconds" and "  <dataset>: box 0.xxx | box+point 0.xxx | point 0.xxx over n structures"
#   The tqdm progress bar is in the .error log.
# Expected time: ~6 GPU-h (0.19 s per structure in job 30271404; ~111,000 structures).
#SBATCH --job-name=sam-bp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_rocm
#SBATCH --gres=gpu:mi355x:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/sam_bp_all/logs/slurm-sam-bp-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/sam_bp_all/logs/slurm-sam-bp-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/user/s4670484/SAMRI}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

ROOT=/scratch/user/s4670484
OUT=$ROOT/Eval_results/sam_bp_all
CKPT=$ROOT/pretrained_ckpt/sam_vit_b_01ec64.pth
NAME=sam_vit_b_01ec64          # test_vis.py names the record after the checkpoint
SEED=0
[ -f "$CKPT" ] || { echo "Missing $CKPT"; exit 1; }
[ -f evaluation/test_vis.py ] || { echo "Not in the repo root: $(pwd)"; exit 1; }
python -c "import sys, torch; print(sys.executable, torch.__version__, torch.version.hip); \
assert torch.cuda.is_available(), 'no GPU visible'; print('device:', torch.cuda.get_device_name(0))" || exit 1
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)"

summary() {  # <record file> : median Dice per prompt for the datasets in one record
python - "$1" <<'PY'
import pickle, sys
import numpy as np
for ds, rows in pickle.load(open(sys.argv[1], "rb")).items():
    med = {k: np.nanmedian(np.array([v for r in rows for v in r[k]], float)) for k in ("b_dice", "bp_dice", "p_dice")}
    n = sum(len(r["labels"]) for r in rows)
    print(f"  {ds}: box {med['b_dice']:.3f} | box+point {med['bp_dice']:.3f} | point {med['p_dice']:.3f} over {n} structures")
PY
}

FAILED=0
for SET in SAMRI_train_test Zeroshot; do
    DATA=$ROOT/Datasets/$SET
    [ -d "$DATA" ] || { echo "Missing $DATA"; exit 1; }
    mapfile -t DSS < <(for d in "$DATA"/*/; do [ -d "$d/testing" ] && basename "$d"; done | sort)
    echo; echo "$SET: ${#DSS[@]} datasets with a testing split"
    for DS in "${DSS[@]}"; do
        DIR=$OUT/$SET/$DS
        if [ -f "$DIR/DONE" ]; then echo "$SET / $DS: done - skipping"; continue; fi
        echo; echo "===== $SET / $DS ====="
        mkdir -p "$DIR"
        # same as: python -m evaluation.test_vis ..., with the random point seeded first
        python - "$DATA/" "$DS" "$CKPT" "$DIR/" "$SEED" <<'PY'
import random, runpy, sys
import numpy as np, torch
data, ds, ckpt, save, seed = sys.argv[1:6]
random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
sys.argv = ["test_vis", "--test-image-path", data, "--datasets", ds, "--ckpt-path", ckpt,
            "--save-path", save, "--model-type", "vit_b", "--device", "cuda"]
runpy.run_module("evaluation.test_vis", run_name="__main__")
PY
        if [ $? -eq 0 ] && [ -s "$DIR/$NAME" ]; then
            summary "$DIR/$NAME" && touch "$DIR/DONE"
        else
            echo "!!!!! FAILED: $SET / $DS (resubmit to retry it)"; FAILED=$((FAILED + 1))
        fi
    done
done

# ---- one table of every structure, and the per-dataset medians -----------------------------------------
echo; echo "===== all datasets ====="
python - "$OUT" "$NAME" <<'PY'
import csv, glob, os, pickle, sys
import numpy as np
out, name = sys.argv[1:3]
keys = ["b_dice", "bp_dice", "p_dice", "b_hd", "bp_hd", "p_hd", "b_msd", "bp_msd", "p_msd", "pixel_count", "area_percentage"]
rows_out, per_ds = [], []
for done in sorted(glob.glob(os.path.join(out, "*", "*", "DONE"))):
    d = os.path.dirname(done)
    test_set = os.path.basename(os.path.dirname(d))
    for ds, rows in pickle.load(open(os.path.join(d, name), "rb")).items():
        first = len(rows_out)
        for r in rows:
            img = r["img_name"][0] if isinstance(r["img_name"], (list, tuple)) else r["img_name"]
            for i, lab in enumerate(r["labels"]):
                rows_out.append([test_set, ds, os.path.basename(str(img)), float(lab)] + [float(r[k][i]) for k in keys])
        a = np.array([x[4:7] for x in rows_out[first:]], float)
        per_ds.append([test_set, ds, len(a)] + [round(float(v), 4) for v in np.nanmedian(a, axis=0)])
with open(os.path.join(out, "sam_vitb_instances.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["test_set", "dataset", "image", "label"] + keys); w.writerows(rows_out)
with open(os.path.join(out, "sam_vitb_dataset_medians.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["test_set", "dataset", "structures", "box", "box+point", "point"]); w.writerows(per_ds)
print(f"{'test set':18s} {'dataset':34s} {'n':>6s}  {'box':>6s} {'box+pt':>6s} {'point':>6s}")
for t, ds, n, b, bp, p in per_ds:
    print(f"{t:18s} {ds:34s} {n:6d}  {b:6.3f} {bp:6.3f} {p:6.3f}")
print(f"\n{len(per_ds)} datasets, {len(rows_out)} structures -> {out}/sam_vitb_instances.csv, sam_vitb_dataset_medians.csv")
PY
echo; echo "finished: $(date); datasets failed: $FAILED"
[ "$FAILED" -eq 0 ]
