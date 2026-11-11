#!/bin/bash
# One-off, on a Bunya LOGIN node (needs internet, ~5 min): a small venv on top of samri-mi355x with nnU-Net v2.7.0.
# The venv sees samri-mi355x's packages (torch 2.13.0+rocm7.2 etc.) and adds only what nnU-Net needs, so the
# shared environment is not modified. torch, torchvision and numpy are pinned to the versions already installed.
#   cd /scratch/user/s4670484/SAMRI && bash embedding_assess/setup_nnunet_env.sh
# Expected last line: "nnU-Net 2.7.0 ready: torch 2.13.0+rocm7.2 (hip ...), numpy ..."
set -euo pipefail
module load anaconda3/2023.09-0
source "$EBROOTANACONDA3/etc/profile.d/conda.sh"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
conda activate /scratch/user/s4670484/conda/samri-mi355x
VENV=/scratch/user/s4670484/conda/nnunet-mi355x-venv
[ -x "$VENV/bin/python" ] || python -m venv --system-site-packages "$VENV"
CONS=$(mktemp)
"$VENV/bin/python" - > "$CONS" <<'EOF'
import importlib.metadata as md  # installed distribution versions, as pip sees them
print("\n".join(f"{p}=={md.version(p)}" for p in ("torch", "torchvision", "numpy")))
EOF
echo "pinned:"; cat "$CONS"
"$VENV/bin/pip" install -c "$CONS" "nnunetv2==2.7.0"
rm -f "$CONS"
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
import numpy, SimpleITK, torch, nnunetv2  # noqa: F401
print(f"nnU-Net {md.version('nnunetv2')} ready: torch {torch.__version__} (hip {torch.version.hip}), numpy {numpy.__version__}")
EOF
