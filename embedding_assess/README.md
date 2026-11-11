# embedding_assess — does SAM's image encoder need MRI fine-tuning?

Experiments behind the SAMRI design choice of fine-tuning only the mask decoder.
They compare **frozen** image encoders on the same MRI slices:

| encoder | what it is |
|---|---|
| `sam_vit_b` | SAM ViT-B, the encoder SAMRI uses (never trained on MRI) |
| `medsam_vit_b` | MedSAM's encoder: the same ViT-B, fully fine-tuned on medical images |
| `sam_vit_h` | SAM ViT-H, the largest SAM encoder (claim: size does not matter) |
| `random_vit_b` | ViT-B with random weights (untrained reference — not a floor, see below) |
| `pixels` | no encoder: the 16×16 pixel patch behind each token (probe baseline) |

Every encoder is run under **both** preprocessing pipelines, so encoders can be
compared on identical inputs (the 2×2 design):

| pipeline | intensity | geometry |
|---|---|---|
| `sam` | per-slice min–max → 0–255, ImageNet mean/std | resize longer side to 1024, zero-pad (SAMRI's stored embeddings) |
| `medsam` | MR percentile clip 0.5–99.5 → 0–255 → [0, 1] | stretch to 1024×1024 (MedSAM inference) |

## Questions and where they are answered

1. **How far did MedSAM's fine-tuning move the encoder?** `cka.py`: layer-wise linear
   CKA between SAM ViT-B and MedSAM on identical inputs, read against a noise ceiling
   (SAM vs SAM with tiny input noise), chance (each image paired with a random other image,
   token order shuffled, ~0) and an untrained
   ViT-B. The untrained network is NOT a floor: two randomly initialised ViT-Bs score
   ~0.99 linear CKA on MRI slices, because random networks largely preserve input
   geometry and linear CKA ignores rotations. So CKA alone is weak evidence; the
   probes carry the main argument. Token subsets: all sampled tokens, and tokens
   inside the GT boxes.
2. **Is what box-prompted segmentation needs already in the frozen embedding?**
   `probe.py`: probes that get the box but no class label (SAMRI never needs to
   know *what* a structure is; the box says *where*):
   - `auc_proto` — no training; prototype separability of target vs background tokens in the box
   - `linear` / `mlp` — one tiny head shared by all datasets, trained on train slices, tested on
     test slices of the trained datasets and on the zero-shot datasets
   - Dice is computed on native pixels and reported against `dice_ceiling` (the best a 64×64 token
     map can do) and `dice_box` (the box alone)
   - paired comparisons against SAM ViT-B on identical inputs, plus SAM ViT-B in the `sam` pipeline
     vs MedSAM in the `medsam` pipeline (each encoder as it is deployed); each is tested over
     instances (optimistic) and over per-dataset medians (the test to quote)
3. **Does encoder size matter (claim 1)?** ViT-H vs ViT-B rows in both outputs.

## Run order (Bunya)

```bash
# 1. CPU node, a few minutes: fixed slice sample + token geometry
python embedding_assess/build_manifest.py \
    --trained-root /scratch/user/s4670484/Datasets/SAMRI_train_test/ \
    --zeroshot-root /scratch/user/s4670484/Datasets/Zeroshot/ \
    --work-dir /scratch/user/s4670484/Eval_results/embedding_research/
mkdir -p /scratch/user/s4670484/Eval_results/embedding_research/logs

# 2. GPU job: all feature sets, then CKA and probes
sbatch embedding_assess/run_embedding_experiments.sh
```

Default sample: 50 training + 25 testing slices per trained dataset, 25 testing slices per
zero-shot dataset (`--n-train`, `--n-test`). `build_manifest.py` prints the disk needed.
Restrict to a subset of datasets with `--datasets NAME [NAME ...]` (e.g. square, already
0–255 datasets where the SAM and MedSAM pipelines give near-identical inputs); use a
separate `--work-dir` per sample so results do not mix.

**First minutes of the job** — check the `.output` log for:
- `device: ...MI300X...` — GPU visible in the right environment
- `[sanity] sam_vit_b own decoder ... mean Dice` ≈ 0.7–0.9 — preprocessing and checkpoint are right
- `[check] ... cosine 1.0000` — SAM features match SAMRI's stored training embeddings
- later `[sanity] medsam_vit_b ... medsam pipeline` ≈ 0.7–0.9

Each finished condition writes `features/<cond>/DONE` and is skipped on resubmission.

## Outputs (`<work-dir>/`)

| file | content |
|---|---|
| `manifest.csv`, `geometry_{sam,medsam}.npz` | slice sample; per-structure token coverage and boxes |
| `features/<encoder>__<pipeline>[__noise1]/` | `neck.npy` (N×256×64×64 fp16), sampled layer tokens for CKA, `meta.json` |
| `results/cka_summary.txt`, `cka_results.csv`, `cka_layers.png` | CKA headline first, then by layer; the plot's lower row shows 1 − CKA on a log axis |
| `results/probe_summary.txt`, `probe_instances.csv` | probe headline first, paired comparisons, per dataset |

## Swapped-encoder decoder test (`run_decoder_experiments.sh`)

Trains identical SAM mask decoders with SAMRI's recipe (AdamW, weight decay 0.1,
Dice + 10×Focal, box jitter ±10 px) on frozen encoder output, using the feature sets
above, so the only thing that changes between arms is the encoder.

| arm id | encoder → decoder start | training data |
|---|---|---|
| `sam-enc_sam-dec_small` | SAM ViT-B → SAM decoder | train slices of the 11 clean datasets (minus 10% validation) |
| `medsam-enc_sam-dec_small` | MedSAM encoder → SAM decoder | same |
| `medsam-enc_medsam-dec_small` | MedSAM encoder → MedSAM decoder | same |
| `sam-enc_sam-dec_all11` | SAM ViT-B → SAM decoder | all training slices of the same 11 datasets (encoded from the images at the start, kept in GPU memory) |

References with no training: SAM ViT-B with its original decoder, MedSAM end to end,
and the released SAMRI checkpoints (36 datasets). Test: tight box, Dice on native
pixels, on the in-domain test slices and on every test slice of the zero-shot datasets.

- **Q1, does the encoder matter?** `medsam-enc_sam-dec_small` − `sam-enc_sam-dec_small` (3 seeds each)
- **Q2, does more training data help zero-shot?** small → all11 → released SAMRI, on the zero-shot sets

Steps (each skipped on resubmission once done): zero-shot manifest + features →
learning-rate pilot (1e-5 vs 1e-4, saved to `chosen_lr.txt`) → 3 small arms × 3 seeds →
all11 arm → `eval_decoder.py`. Outputs in `decoder_swap/`: `runs/<arm>/seed<k>/`
(`best_decoder.pt`, `curve.csv`, `meta.json`) and `results/eval_summary.txt`
(headline first, then paired comparisons, then per dataset) with `eval_instances.csv`.

### LoRA arms (`run_lora_experiments.sh`)

LoRA rank 4 on the query and value projections, all original weights frozen, same data, recipe, validation
slices and test sets as the arms above:

| arm id | what is trained | training data | script |
|---|---|---|---|
| `sam-enc_sam-dec-lora_small` | LoRA on the SAM decoder (frozen encoder) | small data | `train_decoder.py` (stored features) |
| `sam-enc_sam-dec-lora_all11` | LoRA on the SAM decoder (frozen encoder) | all training slices | `train_decoder.py` (encodes once) |
| `sam-enc-lora_sam-dec-lora_small` | LoRA on SAM encoder and decoder | small data | `train_lora.py` (images every step) |
| `sam-enc-lora_sam-dec-lora_all11` | LoRA on SAM encoder and decoder | all training slices, 8,192 per epoch | `train_lora.py` |

Decoder-LoRA weights are saved merged into a plain decoder, so `eval_decoder.py` loads them like the
others. The encoder-LoRA arms change the encoder output, so `train_lora.py` evaluates them itself and
`eval_decoder.py --extra-instances` merges their `eval_instances.csv` into one summary (`results_lora/`).
Each small-data arm runs a learning-rate trial (decoder LoRA 1e-4 / 5e-4 / 1e-3; encoder + decoder LoRA
1e-4 / 5e-4); the winning trial run is the seed-0 run.

### Data size, data variety and compute (`run_curve_experiments.sh`, `run_final_tables.sh`)

Questions: does decoder-only performance keep rising with more slices per dataset; at a fixed number of
slices, does training on more datasets (more structures) help zero-shot; and is more data a better use of a
limited GPU budget than encoder training?

| run ids | what trains | data |
|---|---|---|
| `sam-enc_sam-dec_n<N>` | SAM decoder (frozen encoder) | N = 15, 50, 150, 500, 1,500, 5,000 slices per dataset, seeds 0-2 |
| `sam-enc_sam-dec-lora_n<N>` | LoRA on the SAM decoder | same |
| `sam-enc_sam-dec_k<k>-draw<d>` | SAM decoder | k = 1, 3, 6, 11 datasets, 1,500 slices in total, 3 draws |
| `sam-enc-lora_sam-dec-lora_n1500` (optional) | LoRA on encoder and decoder | 1,500 per dataset, same slices as `sam-enc_sam-dec_n1500` seed 0 |

`data_pools.py` builds the shared slice pool (all training slices minus the validation slices, at most
5,000 per dataset). Sizes are nested within a seed, and seed 0 of every arm sees the same slices.
`train_decoder_curve.py` encodes the pool once and trains every run from GPU memory. `analyse_curve.py`
writes `data_size.csv`, `variety.csv`, `compute.csv` and `summary.txt`. GPU-hours include the one-off
encoding for decoder-only arms, and the variety table splits the clean test sets into datasets seen in
training and datasets left out. Full fine-tuning was left out of the study (Nov 2026); `train_lora.py`
still has the mode.

### Training on a similar dataset (`run_knee_experiments.sh`, `analyse_knee.py`)

Question: does training on a dataset similar to an unseen one raise zero-shot Dice on it? MSK_T2, MSK_PD
and MSK_FLASH are knee MRI in three contrasts with the same six targets (femur, tibia and patella, bone
and cartilage).

| run ids | what trains | data |
|---|---|---|
| `sam-enc_sam-dec_n<N>` (existing) | SAM decoder (frozen encoder) | the 11 clean datasets, N = 150, 500 per dataset, seeds 0-2 |
| `sam-enc_sam-dec_n<N>_plus-MSK_T2` | SAM decoder (frozen encoder) | the same 11 × N slices + N slices of MSK_T2 |

Seed k of a plus run uses exactly the 11-dataset slices of the existing run with seed k (MSK_T2 has its
own pool and seeds in `data_pools.extra_pool`), and validation stays on the same 165 slices, so
plus − base isolates MSK_T2. New test set (`embedding_research_knee/`): every test slice of MSK_PD and
MSK_FLASH (unseen by all models) and MSK_T2 (seen only by the plus runs; a check that they learned it).
Every existing model is also tested on it: `eval_decoder.py --extra-test knee=<dir>` for decoder-only
models, `train_lora.py --eval-only knee=<dir>` for the encoder + decoder LoRA arms. There is no control
arm with one more non-knee dataset, so a gain shows that MSK_T2 helps, not that a similar dataset helps
more than any extra dataset would. Outputs in `decoder_swap/results_knee/`: `eval_summary.txt`,
`eval_instances.csv` and `tables/knee_summary.txt` (plus vs base headline, per structure type, every
system on the knee sets).

## Files

- `emb_common.py` — pipelines, geometry (mask → token coverage, token map → native pixels), model loading
- `build_manifest.py` → `extract_features.py` → `cka.py`, `probe.py`
- `run_embedding_experiments.sh` — Slurm job (1 × MI300X, `--qos=gpu`, account `a_ai_collab`)
- `medsam_check.py` — separate CPU check of MedSAM preprocessing on the test slices
- `decoder_lib.py`, `train_decoder.py`, `eval_decoder.py`, `run_decoder_experiments.sh` — swapped-encoder decoder test
- `train_lora.py`, `run_lora_experiments.sh` — LoRA arms (decoder LoRA via `train_decoder.py`)
- `data_pools.py`, `train_decoder_curve.py`, `analyse_curve.py`, `run_curve_experiments.sh`, `run_final_tables.sh` — data size, variety and compute
- `run_knee_experiments.sh`, `analyse_knee.py` — training on a similar dataset (11 datasets + MSK_T2, knee test set)
