#!/bin/bash --login
# nnU-Net comparison, step 1 (CPU only, ~1 h): write the 12 nnU-Net datasets (the 11 clean datasets and MSK_T2,
# each its own full training split, with its patient-wise validation split as nnU-Net's fold 0) and their test
# images, then nnU-Net's own planning and 2D preprocessing. About 150k training + validation slices in total, so
# roughly a million files and a few tens of GB under Eval_results/nnunet_knee - check the scratch quota first.
#
# BEFORE: bash embedding_assess/setup_nnunet_env.sh (once, login node), and step 1 of run_knee_experiments.sh
# has written the knee manifest (the MSK_T2/PD/FLASH test rows). Submit from the repo root:
#   cd /scratch/user/s4670484/SAMRI && sbatch embedding_assess/run_nnunet_prepare.sh
# Re-running skips datasets already written (DONE files) and preprocesses only datasets whose nnU-Net
# preprocessed folder is missing or incomplete.
#
# Lesion fix (second run, ~15 min): the three lesion datasets store one label number per lesion; the first run
# made each number a class. Re-running this script writes them again with one lesion class (array indices
# 12-14: ISLES2022_ADC_onelabel, ISLES2022_DWI_onelabel, QIN-PROSTATE-Lesion_onelabel) and preprocesses only
# those three; the 12 first-run datasets are left as they are. Then train: sbatch --array=12-14 run_nnunet_train.sh
#
# CHECK the .output log:
#   "[k] DatasetNNN_<name>: done - skipping" for the 12 first-run datasets (0-11)
#   "[12] Dataset913_ISLES2022_ADC_onelabel: 2345 training + 315 validation slices (fold 0); one label for every
#    lesion; labels (original -> nnU-Net ...): 2->1 ..., 3->1 ..." (all -> 1), same for 13 and 14
#   "array index 12..14", "planning and 2D preprocessing: 913 914 915", nnU-Net's "2D U-Net configuration" for each, no
#   integrity errors
#SBATCH --job-name=samri-nnunet-prep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=general
#SBATCH --account=a_ai_collab
#SBATCH -o /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-nnunet-prep-%j.output
#SBATCH -e /scratch/user/s4670484/Eval_results/decoder_swap/logs/slurm-nnunet-prep-%j.error
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhao.wang1@uq.edu.au

module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
source /scratch/user/s4670484/conda/nnunet-mi355x-venv/bin/activate
python -c "import nnunetv2, SimpleITK, nibabel" \
    || { echo "nnU-Net venv missing (python: $(which python)) - run setup_nnunet_env.sh first"; exit 1; }
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-/scratch/user/s4670484/SAMRI}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOT=/scratch/user/s4670484
NN=$ROOT/Eval_results/nnunet_knee
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))
export nnUNet_def_n_proc=$WORKERS
echo "repo: $(pwd)  commit: $(git rev-parse --short HEAD 2>/dev/null)  python: $(which python)"
python embedding_assess/nnunet_prepare.py --traintest "$ROOT/Datasets/SAMRI_train_test" \
    --indomain-dir "$ROOT/Eval_results/embedding_research_clean" \
    --knee-dir "$ROOT/Eval_results/embedding_research_knee" --nnunet-root "$NN" --workers "$WORKERS" \
    || { echo "!!!!! FAILED: nnunet_prepare"; exit 1; }

export nnUNet_raw=$NN/raw nnUNet_preprocessed=$NN/preprocessed nnUNet_results=$NN/results
# preprocess only datasets whose preprocessed 2D folder is missing or holds fewer cases than training +
# validation slices (nnU-Net writes one .pkl per case, last)
PRE=$(python - "$NN" <<'PY'
import glob, json, os, sys
nn = sys.argv[1]
todo = []
for e in json.load(open(os.path.join(nn, "datasets.json"))):
    n = len(glob.glob(os.path.join(nn, "preprocessed", e["folder"], "nnUNetPlans_2d", "*.pkl")))
    if n < e["n_train"] + e["n_val"]:
        todo.append(str(e["id"]))
    print(f"  {e['folder']}: {n} of {e['n_train'] + e['n_val']} cases already preprocessed")
print("IDS:" + " ".join(todo))
PY
) || { echo "!!!!! FAILED: checking preprocessed folders"; exit 1; }
echo; echo "$PRE" | grep -v '^IDS:'
IDS=$(echo "$PRE" | grep '^IDS:' | cut -c5-)
echo "===== nnU-Net planning and 2D preprocessing: ${IDS:-nothing to do} ====="
if [ -n "$IDS" ]; then
    nnUNetv2_plan_and_preprocess -d $IDS -c 2d -np "$WORKERS" -npfp "$WORKERS" --verify_dataset_integrity \
        || { echo "!!!!! FAILED: nnUNetv2_plan_and_preprocess"; exit 1; }
fi
echo "files under $NN: $(find "$NN" -type f | wc -l), size: $(du -sh "$NN" | cut -f1)"
echo "all steps finished: $(date)"
