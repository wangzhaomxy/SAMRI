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
  --input ./datasets/demoSample/example_img_1.nii.gz \
  --output ./datasets/infer_output\
  --checkpoint ./models/samri_vitb_bp.pth \
  --model-type samri \
  --device mps \
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
## 📊 Training


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
