# 🧠 SAMRI: Segment Anything Model for MRI

**SAMRI** is an MRI-specialized adaptation of [Meta AI’s Segment Anything Model (SAM)](https://segment-anything.com/), designed for accurate and efficient segmentation across diverse MRI datasets.  
By fine-tuning only the **lightweight mask decoder** on **precomputed MRI embeddings**, SAMRI achieves state-of-the-art Dice and boundary accuracy while drastically reducing computational cost.

---

## 🌟 Highlights

- 🧩 **Decoder-only fine-tuning** — freeze SAM’s heavy image encoder and prompt encoder.  
- ⚙️ **Two-stage pipeline** — precompute embeddings → fine-tune decoder.  
- 🧠 **1.1 M MRI pairs** from **36 datasets / 47 tasks** across **10+ MRI protocols**.  
- 🚀 **94% shorter training time** and **96% fewer trainable parameters** than full SAM retraining.  
- 📈 **Superior segmentation** on small and medium structures, with strong zero-shot generalization.  
- 🖼️ Supports **box, point, and box + point prompts**.  

---

![SAMRI Architecture](docs/fig_samri_architecture.png)  
*Figure 1. Overview of SAMRI: frozen image encoder and prompt encoder, lightweight decoder fine-tuning.*

---

## 🧭 Overview

**SAMRI** adapts SAM for the MRI domain by leveraging SAM’s strong visual representations while tailoring the decoder to medical structures and contrasts.  
The approach:
1. **Precomputes embeddings** using SAM ViT-B encoder on 2D MRI slices.  
2. **Fine-tunes only the mask decoder** with a hybrid focal–Dice loss for domain adaptation.  

This lightweight strategy allows SAMRI to train efficiently on a **single GPU** or **multi-GPU clusters** (e.g., H100 x 8), while maintaining robust accuracy across unseen datasets and imaging protocols.

---
## 🛠️ Installation (SAMRI)

This section helps you go from **zero to a runnable environment** for SAMRI. It includes optional prerequisites, a reproducible Conda setup, and a brief explanation of how dependency installation works.



### 🧩 Step 0 — Prerequisites (Optional but recommended)
SAMRI requires **Python ≥ 3.10** and **PyTorch ≥ 2.2** (CUDA or ROCm recommended).  
Use a package manager like **Conda** to isolate dependencies per project.

- Download [**Anaconda**:arrow_upper_right:](https://www.anaconda.com/download)
- Download [**Miniconda (lightweight)**:arrow_upper_right:](https://docs.conda.io/en/latest/miniconda.html)


Verify Conda is available:
```bash
conda --version
```

### 🔁 Step 1 — Create and activate a fresh environment
If you already have a base environment:

```bash
conda create -n samri python=3.10 -y
conda activate samri
```

### 🧪 Step 2 - Install PyTorch

Please install the correct [PyTorch:arrow_upper_right:](https://pytorch.org) version according to your operating system, package manager, language, and compute platform.
**Note:** This project has been verified on **PyTorch 2.2.0.**

### 🧰 Step 3 — Clone the Repository and install dependencies
```bash
git clone https://github.com/wangzhaomxy/SAMRI.git
cd SAMRI
pip install .
```

### ✅ Step 4 — Verify Your Setup
Run a quick import test in the command line:
```bash
python -c "import torch, nibabel; print('SAMRI environment ready! Torch:', torch.__version__)"
```

If it prints without errors, your environment is correctly configured.

---
## 🚀 Quick Start (Inference & Visualization only)

This project ships two entry points for running SAMRI on your data:

- **CLI**: `inference.py` — fast, scriptable inference and saving of masks/PNGs
- **Notebook**: `infer_step_by_step.ipynb` — interactive, cell-by-cell walkthrough

Both files live in the **repo root**.

---

### 1️⃣ Inference (CLI) — `inference.py`

Run SAM/SAMRI on a single NIfTI (`.nii/.nii.gz`) **or** standard image (`.png/.jpg/.tif`) and save the predicted mask.

**Basic usage**
```bash
python inference.py \
  --input ./data/sample_case01.nii.gz \
  --output ./Inference_results/ \
  --checkpoint ./models/samri_vitb_bp.pth \
  --model-type samri \
  --device cuda \    # Alter with "mps" for apple silicon
  --box X1 Y1 X2 Y2\
  --point X Y \
  --no-png False

```

**CLI arguments (from `inference.py`)**
- `--input, -i` (required): path to `.nii/.nii.gz` **or** `.png/.jpg/.tif`
- `--output, -o` (required): output folder where results are written
- `--checkpoint, -c` (required): path to SAM/SAMRI checkpoint (`.pth`)
- `--model-type` (default: `vit_b`): one of `vit_b | vit_h | samri` (`samri` maps to ViT-B backbone)
- `--device` (default: `cuda`): e.g., `cuda`, `cpu` (or `mps` on Apple Silicon if available)
- `--box X1 Y1 X2 Y2` (required): bounding box prompt (pixels)
- `--point X Y` (optional): foreground point prompt (pixels)
- `--no-png` (flag): if set, do **not** save PNG; only `.nii.gz` mask is written

**Outputs**
- `<name>_seg_.nii.gz` — predicted mask saved as NIfTI with shape `[1, H, W]`
- `<name>_seg_.png` — (unless `--no-png`) grayscale binary mask PNG. 

**Example**
```bash
python inference.py \
  --input ./user_data/Datasets/demoSample/example_img_1.nii.gz \
  --output ./user_data/Datasets/infer_output \
  --checkpoint ./user_data/pretrained_ckpt/samri_vitb_bp.pth \
  --model-type samri \
  --device mps \ # or "cuda"
  --box 115 130 178 179\
  --point 133 172
```

> The input must be a 2D image. It is automatically normalized to 8-bit and converted to RGB to align with SAM’s internal preprocessing. The expected NIfTI file shape is (1, H, W) or (H, W), with (H, W, 1) also supported via automatic squeezing. The image input accepts dimensions in any of the following forms: H×W, H×W×1, H×W×3, or H×W×4.
---

### 2️⃣ Visualize step-by-step (Notebook) — `infer_step_by_step.ipynb`

Use the notebook to experiment with prompts and visualize each stage.

**Open** `./infer_step_by_step.ipynb` and set the cell parameters:
```python
# --- User configuration ---
INPUT_PATH = "/path/to/your/input.nii.gz"   # or .png/.jpg
OUTPUT_DIR = "./Notebook_Visualization"
CHECKPOINT = "./checkpoints/samri_decoder.pth"  # SAM / SAMRI checkpoint
MODEL_TYPE = "samri"  # 'vit_b' | 'vit_h' | 'samri'
DEVICE = "cuda"       # 'cuda' | 'cpu' | 'mps'

# Optional prompts (pixel coords)
BOX = [30, 40, 200, 220]   # or None
POINT = [120, 140]         # or None
SAVE_PNG = True            # also write PNG next to the NIfTI
```

**Then run cells** to:
1. Load & normalize the input (NIfTI or image)
2. Configure optional **box/point** prompts
3. Run SAMRI inference
4. Save: `<name>_seg_.nii.gz` (+ optional `<name>_seg_.png`)
5. Display publication-friendly overlays/contours inside the notebook

> The notebook uses the same image preparation and I/O utilities as the CLI, ensuring identical masks for matching inputs and prompts.
---
## 🧑‍🏫 Training the Model

This section covers **end‑to‑end training** of SAMRI’s decoder on precomputed SAM embeddings. The workflow is lightweight:
1) **Prepare data** → 2) **Precompute embeddings** → 3) **Train decoder** → 4) (Optional) **Evaluate/visualize**.

> SAMRI freezes SAM’s image encoder and fine‑tunes only the **mask decoder** using a Dice+Focal loss.

---

### 📂 1) Prepare Your Data

Organize datasets as study folders with images and masks. Patient-wise split the training/validation/testing samples. 
Examples:
```
./user_data/Datasets/SAMRI_train_test
├── dataset_A/
│   ├── training/
│   │   ├──example1*_img_*.nii.gz
│   │   ├──example1*_seg_*.nii.gz    # matching file names (binary/label masks)
│   │   └──...
│   ├── validation/         # .nii.gz
│   │   └──...
│   └── testing/
│   │   └──...          
├── dataset_B/
│   ├── training/
│   ├── validation/         # .nii.gz
│   └── testing/   
└── ...
```

> * Masks should align with images in shape and orientation. 
> * For 3D NIfTI, training is typically on **2D slices**.
> * The image and mask files should be organized in the same folder with different keys: **"\_img_\"** for images, and **"\_seg_\"** for masks, respectively. Other part of the name should be the same or in the same order after being sorted.

Optional split files:
```
splits/
  train.txt   # each line = relative path to an image
  val.txt
  test.txt
```

---

### ⚙️ 2) Precompute Image Embeddings
Use SAM ViT‑B to compute and cache image embeddings (saves training time & memory).

```bash
python preprocess/precompute_embeddings.py   --data_dir /data/SAMRI_train_test   --save_dir ./embeddings   --batch_size 16   --num_workers 8
```
**Key args**
- `--data_dir` : root folder with datasets
- `--save_dir` : output folder for `.pt` or `.npy` embeddings
- `--batch_size` : embedding mini‑batch size (per process if using DDP)
- `--num_workers` : DataLoader workers (tune to avoid CPU/IO bottlenecks)

> Tip: If you see **OOM** or heavy swapping, lower `--batch_size` or `--num_workers` (e.g., 2–8).

---

### 🎯 3) Train the Decoder

#### Single‑GPU
```bash
python train_decoder.py   --embedding_dir ./embeddings   --epochs 30   --batch_size 16   --lr 1e-4   --save_dir ./checkpoints
```

#### Multi‑GPU (same node, PyTorch DDP)
```bash
torchrun --nproc_per_node=8 train_decoder.py   --embedding_dir ./embeddings   --epochs 30   --batch_size 16   --lr 1e-4   --save_dir ./checkpoints
```

#### SLURM (example for Bunya MI300X nodes)
```bash
# sbatch train_samri_ddp.sbatch
#!/bin/bash
#SBATCH -J samri_ddp
#SBATCH -A <your_account>
#SBATCH -p mi300x
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
# load your conda and env
source ~/.bashrc
conda activate samri

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NCCL_BLOCKING_WAIT=1  # easier debugging
export NCCL_P2P_DISABLE=1          # sometimes helps on ROCm

srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE train_decoder.py   --embedding_dir ./embeddings   --epochs 30   --batch_size 16   --lr 1e-4   --save_dir ./checkpoints
```

**Common args (from `train_decoder.py`)**
- `--embedding_dir` : path to precomputed embeddings
- `--epochs` : number of training epochs
- `--batch_size` : per‑process batch size (effective = batch_size × world_size)
- `--lr` : learning rate (e.g., 1e‑4)
- `--save_dir` : where to write checkpoints (`.pth`) & logs
- `--amp` : (flag) enable mixed precision fp16/bf16 if supported
- `--seed` : seed for reproducibility
- `--val_every` : validate every N steps/epochs (if supported)

**Resuming**
```bash
python train_decoder.py   --embedding_dir ./embeddings   --epochs 30   --batch_size 16   --lr 1e-4   --save_dir ./checkpoints   --resume ./checkpoints/last.pth
```

---

### 🧪 4) Evaluate / Visualize (Optional)
After training, use the inference tools to inspect results:

```bash
# CLI visualization (PNG + NIfTI)
python inference.py   --input /path/to/case001.nii.gz   --output ./Inference_results/case001   --checkpoint ./checkpoints/best.pth   --model-type samri   --box 30 40 200 220
```

Or run the notebook:
```bash
jupyter lab
# open: ./infer_step_by_step.ipynb
```

---

### 📝 Reproducibility Checklist
- Fix seeds: `--seed 42` (and set `torch.backends.cudnn.deterministic=True` if applicable)
- Log environment: `pip freeze > logs/requirements.txt`
- Save CLI args / config: auto‑dump to `save_dir/args.json`
- Track git commit: write `git rev-parse HEAD` to logs

---

### 🧯 Troubleshooting
- **CUDA/ROCm OOM**: lower `--batch_size`; reduce `num_workers`; enable `--amp`
- **Slow data loading**: set `--num_workers 4..8`, `pin_memory=True` (if CUDA)
- **DDP hangs**: check `MASTER_ADDR/PORT`; try `export NCCL_P2P_DISABLE=1` on ROCm
- **Validation mismatch**: confirm same preprocessing/normalization as training

---

### 🔎 Notes on Backends
- **PyTorch/CUDA**: install a build matching your CUDA version
- **ROCm (AMD MI300X/MI210)**: use ROCm PyTorch wheels; NCCL flags above may help
- **Apple Silicon (MPS)**: training is possible, but performance is limited compared to CUDA/ROCm

---

### 📦 Expected Artifacts
```
checkpoints/
  best.pth          # best validation metric
  last.pth          # last epoch
  logs.json         # training curves/metrics
embeddings/
  dataset_A/*.pt    # precomputed features
  dataset_B/*.pt
```
If you want, you can pin typical hyper‑parameters per dataset in `configs/*.yaml` and pass `--config configs/amosmr.yaml` (if your script supports it).

---
## 📊 Evaluation

SAMRI is evaluated using:
- **Dice Similarity Coefficient (DSC)**  
- **Hausdorff Distance (HD)**  
- **Mean Surface Distance (MSD)**  

We further group structures by **relative size**:  
- *Small* (< 0.5%), *Medium* (0.5–3.5%), *Large* (> 3.5%)  
and apply **Wilcoxon signed-rank tests** to assess significance.

![Results](docs/fig_samri_results.png)  
*Figure 2. Dice comparison between SAM ViT-B, MedSAM, and SAMRI across object-size bins.*

---

## 🧠 Dataset Overview

SAMRI is trained on a curated **1.1 million MRI image–mask pairs** from **36 public datasets** (47 segmentation tasks) spanning over **10 MRI sequences** (T1, T2, FLAIR, DESS, TSE, etc.).

| Category | Example Datasets | Approx. Pairs |
|-----------|------------------|---------------|
| **Brain** | BraTS, ISLES, MSD_Hippocampus | 420 K |
| **Abdomen** | AMOSMR, HipMRI | 260 K |
| **Knee** | MSK_T2, OAI, DESS | 210 K |
| **Thorax** | Heart, MSD_Heart | 130 K |
| **Others** | Prostate, MSK_FLASH | 80 K |

Detailed dataset breakdowns are provided in **Table S1 (Supplementary)**.

---

## 📂 Repository Structure

```
SAMRI/
├── configs/             # Dataset/task configurations (YAML)
├── preprocess/          # Precompute embeddings and data utilities
├── train_decoder.py     # Decoder fine-tuning script
├── infer_results.py     # Inference and visualization pipeline
├── utils/               # Metrics, plotting, and helper functions
├── requirements.txt
└── README.md
```

---

## 🧪 Results Summary

| Model | Trainable Params | Training Time (8× MI300X) | Dice ↑ | HD ↓ | MSD ↓ |
|--------|------------------|----------------------------|--------|------|------|
| SAM ViT-B (zero-shot) | 0 % | — | Baseline | — | — |
| MedSAM | 100 % | > 600 h | Moderate | — | — |
| **SAMRI (Ours)** | **4 %** | **76 h** | **↑ Dice, ↓ HD, ↓ MSD** | | |

SAMRI shows the largest gains on **small and medium objects**, consistent with qualitative boundary adherence improvements.

---

## 📘 Citation

If you use SAMRI in your research, please cite:

<!-- ```bibtex
@article{wang2025samri,
  title={SAMRI: Segment Anything Model for MRI},
  author={Wang, Zhao and Chandra, Shekhar and Dai, Wei and others},
  journal={Nature Communications},
  year={2025}
}
``` -->

---

## 📄 License

This repository is released under the **Apache 2.0 License** (or specify otherwise).  
See the [LICENSE](LICENSE) file for details.

---
## 🤝 Acknowledgments

Developed at **The University of Queensland (UQ)**,  
**School of Electrical Engineering and Computer Science (EECS)**.

Special thanks to the **Bunya HPC Team** for infrastructure support.

Built upon **Meta AI’s Segment Anything Model (SAM)**, and inspired by the broader community efforts to adapt SAM to medical imaging.  

We also gratefully acknowledge the **MedSAM team** for pioneering open-source adaptations of SAM for medical images and for releasing code/weights that served as an important baseline and point of comparison.

We thank open-source contributors and the MRI research community for dataset availability.

---

## 📬 Contact

**Zhao Wang**  
School of Electrical Engineering and Computer Science (EECS)  
The University of Queensland, Australia  
📧 zhao.wang1@uq.edu.au

**Shekhar “Shakes” Chandra**
ARC Future Fellow & Senior Lecturer
School of Electrical Engineering and Computer Science (EECS)  
The University of Queensland, Australia  
📧 shekhar.chandra@uq.edu.au
