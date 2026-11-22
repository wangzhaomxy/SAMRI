#!/bin/bash --login
# Rerun the SAM-Med2D baseline on EVERY test set with the fixed input pipeline
# (evaluation/test_comparison_models.py, infer_sam_med2d fixed 14 Nov 2026).
#
# Why: the published SAM-Med2D numbers were near zero because the test code assumed 256x256 slices.
# Larger slices were cropped to their top-left 256x256 corner and the box was sent in native pixel
# coordinates. The fixed code follows SAM-Med2D's own test.py: stretch (or centre-pad) to 256x256,
# box in that frame, mask mapped back to the native size.
#
#   test sets   every dataset folder with a testing/ split under Datasets/SAMRI_train_test ("trained")
#               and Datasets/Zeroshot ("zero_shot")
#   prompt      tight ground-truth box of the whole structure, no jitter (same as every other model)
#   resumable   one dataset per call; a finished dataset is skipped on resubmission
#   output      NEW folder - the old sam_med2d.pkl is not touched:
#               Eval_results/SAMRI_comparison/sam_med2d_fixed/sam_med2d_fixed.pkl
#               (same structure as the old file: {"trained": {dataset: [...]}, "zero_shot": {...}})
#
# BEFORE submitting (once):
#   mkdir -p /scratch/user/s4670484/Eval_results/SAMRI_comparison/sam_med2d_fixed/logs
# Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch evaluation/run_sam_med2d_rerun.sh
# Card: 1 x NVIDIA H100 (partition gpu_cuda) with the sam_med2d conda env (a CUDA build of torch).
# If sbatch rejects the --gres name, list the valid ones with  sinfo -p gpu_cuda -o "%G"  and pass it:
#   sbatch --gres=gpu:<name>:1 evaluation/run_sam_med2d_rerun.sh
# WITHOUT sbatch, on a GPU you already hold (interactive / onBunya desktop session) - same checks, same
# resume, output on screen and in a log file. REPO is needed because SLURM_SUBMIT_DIR there is not the repo:
#   cd /scratch/user/s4670484/SAMRI && REPO=$PWD bash -l evaluation/run_sam_med2d_rerun.sh 2>&1 | tee -a sam_med2d_rerun.log
# On an AMD card (MI300X / MI355X) add  CONDA_ENV=/scratch/user/s4670484/conda/samri-mi355x  before  bash.
# Another conda env (it needs torch, nibabel, scipy, tqdm and cv2):
#   sbatch --export=ALL,CONDA_ENV=/path/to/env evaluation/run_sam_med2d_rerun.sh
#
# FIRST MINUTES - in the .output log:
#   "device: NVIDIA H100 ..."
#   "SAMRI_train_test: N datasets with a testing split" and "Zeroshot: 6 datasets ..."
#   "===== quick check ..." then lines like  labels=[1, 2]  dice=['0.9xx', '0.8xx']
#       -> Dice should be clearly above zero (roughly 0.5-0.95). Near zero on every line means scancel and check.
#   "===== trained / <first dataset> =====" and, when it finishes, "  <dataset>: median Dice 0.xxx over n structures"
#   The tqdm progress bar is in the .error log.
# At the end: a table of median Dice per dataset, new next to the old sam_med2d.pkl.
#SBATCH --job-name=sammed2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/SAMRI_comparison/sam_med2d_fixed/logs/slurm-sammed2d-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/SAMRI_comparison/sam_med2d_fixed/logs/slurm-sammed2d-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
CONDA_ENV=${CONDA_ENV:-/scratch/user/s4670484/conda/sam_med2d}
conda activate "$CONDA_ENV"
python -c "import torch, numpy, nibabel, scipy, tqdm, cv2" \
    || { echo "$CONDA_ENV is missing a package (python: $(which python)) - see the .error log."
         echo "If it is cv2:  python -m pip install opencv-python-headless \"numpy==\$(python -c 'import numpy; print(numpy.__version__)')\""
         exit 1; }
set -uo pipefail
cd "${REPO:-${SLURM_SUBMIT_DIR:-/scratch/user/s4670484/SAMRI}}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# AMD cards (ROCm build of torch) need these; they do nothing on an NVIDIA card
unset ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export MIOPEN_USER_DB_PATH="/dev/shm/miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# ---- job body ----------------------------------------------------------------------------------
ROOT=${ROOT:-/scratch/user/s4670484}
DEVICE=${DEVICE:-cuda}
CKPT=${CKPT:-$ROOT/comparison_ckpt_samri/SAM_Med2D/sam-med2d_b.pth}
OUT=${OUT:-$ROOT/Eval_results/SAMRI_comparison/sam_med2d_fixed}
OLD=${OLD:-$ROOT/Eval_results/SAMRI_comparison/sam_med2d.pkl}      # the published (broken) run, for the table
PARTS=$OUT/parts
FINAL=$OUT/sam_med2d_fixed.pkl
SPLITS="trained:$ROOT/Datasets/SAMRI_train_test zero_shot:$ROOT/Datasets/Zeroshot"
mkdir -p "$PARTS"
[ -f "$CKPT" ] || { echo "Missing $CKPT"; exit 1; }
[ -f evaluation/test_comparison_models.py ] || { echo "Not in the repo root: $(pwd)"; exit 1; }
grep -q "_sam_med2d_inputs" evaluation/test_comparison_models.py \
    || { echo "evaluation/test_comparison_models.py is the OLD version (no _sam_med2d_inputs) - copy the fixed file first"; exit 1; }
if [ "$DEVICE" = cuda ]; then
    python -c "import sys, torch; print(sys.executable, '| python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| hip', getattr(torch.version, 'hip', None)); \
assert torch.cuda.is_available(), 'no GPU visible - wrong conda env for this card? (NVIDIA: sam_med2d, AMD: samri-mi355x)'; print('device:', torch.cuda.get_device_name(0))" || exit 1
fi
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"

datasets_of () {  # <root> : dataset folders that have a testing/ split
    for d in "$1"/*/; do [ -d "${d}testing" ] && basename "$d"; done
}
run_one () {  # <root> <dataset> <save path> [extra args]
    python evaluation/test_comparison_models.py --model sam_med2d --ckpt-path "$CKPT" \
        --dataset-path "$1" --datasets "$2" --save-path "$3" --device "$DEVICE" "${@:4}"
}
part_summary () {  # <part file>
python - "$1" <<'PY'
import pickle, sys
import numpy as np
for split in pickle.load(open(sys.argv[1], "rb")).values():
    for ds, rows in split.items():
        d = np.array([v for r in rows for v in r["dice"]], float)
        print(f"  {ds}: median Dice {np.nanmedian(d):.3f} over {len(d)} structures")
PY
}

for s in $SPLITS; do
    root=${s#*:}
    [ -d "$root" ] || { echo "Missing $root"; exit 1; }
    echo "$(basename "$root"): $(datasets_of "$root" | wc -l | tr -d ' ') datasets with a testing split"
done

# ---- quick check: 2 slices of the first dataset, Dice printed per structure ------------------------
first_root=${SPLITS%% *}; first_root=${first_root#*:}
first_ds=$(datasets_of "$first_root" | head -n 1)
echo; echo "===== quick check: 2 slices of $first_ds (Dice per structure should be clearly above zero) ====="
run_one "$first_root" "$first_ds" "$OUT/quick_check.pkl" --debug 2>&1 | grep -E "labels=|Error|Traceback" \
    || { echo "!!!!! FAILED: quick check did not run cleanly (see any Traceback above and the .error log) - stopping"; exit 1; }

# ---- every dataset, one call each (resumable) -----------------------------------------------------
failed=0
for s in $SPLITS; do
    split=${s%%:*}; root=${s#*:}
    for ds in $(datasets_of "$root"); do
        part="$PARTS/${split}__${ds}.pkl"
        echo; echo "===== $split / $ds ====="
        if [ -f "$part" ]; then
            echo "  already finished - skipping"
        else
            t0=$SECONDS
            if run_one "$root" "$ds" "$part.tmp" | grep -E "Error|Traceback|Results saved"; then
                mv "$part.tmp" "$part" && echo "  $(( SECONDS - t0 )) s"
            else
                echo "!!!!! FAILED: $split / $ds"; rm -f "$part.tmp"; failed=$(( failed + 1 )); continue
            fi
        fi
        part_summary "$part"
    done
done

# ---- merge into one file (old format) and compare with the published run ---------------------------
echo; echo "===== summary: median Dice per dataset, fixed pipeline vs the old sam_med2d.pkl ====="
python - "$PARTS" "$FINAL" "$OLD" "$SPLITS" <<'PY'
import glob, os, pickle, sys
import numpy as np
parts, final, old_path, splits = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
merged, missing = {}, []
for s in splits:
    split, root = s.split(":", 1)
    merged[split] = {}
    for d in sorted(glob.glob(os.path.join(root, "*", "testing"))):
        ds = os.path.basename(os.path.dirname(d))
        p = os.path.join(parts, f"{split}__{ds}.pkl")
        if os.path.isfile(p):
            merged[split][ds] = next(iter(pickle.load(open(p, "rb")).values()))[ds]
        else:
            missing.append(f"{split}/{ds}")
with open(final, "wb") as f:
    pickle.dump(merged, f)
old = pickle.load(open(old_path, "rb")) if os.path.isfile(old_path) else {}
med = lambda rows: float(np.nanmedian(np.array([v for r in rows for v in r["dice"]], float))) if rows else float("nan")
for split, dss in merged.items():
    print(f"  [{split}]")
    print(f"    {'dataset':32s} {'slice side':>10s} {'structures':>10s} {'fixed':>7s} {'old':>7s}")
    new_m, old_m = [], []
    for ds, rows in dss.items():
        side = np.sqrt(np.median([p / a for r in rows for p, a in zip(r["pixel_count"], r["area_percentage"]) if a > 0] or [np.nan]))
        o = med(old.get(split, {}).get(ds, []))
        new_m.append(med(rows)); old_m.append(o)
        print(f"    {ds:32s} {side:10.0f} {sum(len(r['dice']) for r in rows):10d} {new_m[-1]:7.3f} {o:7.3f}")
    if new_m:
        print(f"    {'mean of dataset medians':32s} {'':10s} {'':10s} {np.nanmean(new_m):7.3f} {np.nanmean(old_m) if not np.all(np.isnan(old_m)) else float('nan'):7.3f}")
print(f"\nmerged {sum(len(v) for v in merged.values())} datasets -> {final}")
if not old:
    print(f"(no old file at {old_path}: the 'old' column is empty)")
if missing:
    print("NOT FINISHED (resubmit the same command):", ", ".join(missing))
PY
[ "$failed" -eq 0 ] || echo "!!!!! $failed dataset(s) failed - see above; resubmit to retry them"
echo; echo "finished: $(date)"
