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

## Files

- `emb_common.py` — pipelines, geometry (mask → token coverage, token map → native pixels), model loading
- `build_manifest.py` → `extract_features.py` → `cka.py`, `probe.py`
- `run_embedding_experiments.sh` — Slurm job (1 × MI300X, `--qos=gpu`, account `a_ai_collab`)
- `medsam_check.py` — separate CPU check of MedSAM preprocessing on the test slices
- `decoder_lib.py`, `train_decoder.py`, `eval_decoder.py`, `run_decoder_experiments.sh` — swapped-encoder decoder test
