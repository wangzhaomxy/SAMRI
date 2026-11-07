# -*- coding: utf-8 -*-
"""
Training-slice pools for the data-size and data-variety experiments (train_decoder_curve.py) and the
per-dataset image-trained arms (train_lora.py, data "n<N>").

pool       every training slice of each clean dataset (the training/ folders the in-domain manifest's
           training rows come from), minus the shared validation slices, in a fixed random order per
           dataset (seed 0), cut at POOL_MAX per dataset.
size runs  N slices per dataset = the first N of a seed-specific order of that dataset's pool, so for one
           seed the sizes are nested (the 150 slices are part of the 500, ...), and seed 0 of any arm
           (decoder-only, decoder LoRA, encoder LoRA, full fine-tuning) sees exactly the same slices.
           A dataset with fewer than N pool slices contributes all of them.
variety    k datasets drawn at random (per draw) among those with >= total // k pool slices, each giving
           total // k slices: the total stays fixed while the number of datasets (structures) changes.
"""

import os

import numpy as np

POOL_MAX = 5000  # slices per dataset kept in the pool (BraTS has ~12k per modality, most others fewer)


def training_pool(manifest, val_rows, pool_max=POOL_MAX, seed=0):
    """dataset -> list of (image path, mask path), validation slices removed, in pool order."""
    from build_manifest import list_pairs
    val_paths = {manifest[r]["img_path"] for r in val_rows}
    folders = {r["dataset"]: os.path.dirname(r["img_path"]) for r in manifest if r["split"] == "train"}
    pool, seen = {}, 0
    for i, ds in enumerate(sorted(folders)):
        pairs = list_pairs(folders[ds])[0]
        seen += sum(p[0] in val_paths for p in pairs)
        pairs = [p for p in pairs if p[0] not in val_paths]
        order = np.random.default_rng([seed, i]).permutation(len(pairs))
        pool[ds] = [pairs[j] for j in order[:pool_max]]
    if seen != len(val_paths):
        raise SystemExit(f"only {seen} of {len(val_paths)} validation slices found in the training folders - "
                         "stopping (validation would overlap training)")
    return pool


def size_subset(pool, n, seed):
    """dataset -> pool indices of the first n slices of a seed-specific order (nested across n)."""
    out = {}
    for i, ds in enumerate(sorted(pool)):
        order = np.random.default_rng([1000 + seed, i]).permutation(len(pool[ds]))
        out[ds] = sorted(int(j) for j in order[:n])
    return out


def variety_subset(pool, k, total, draw):
    """dataset -> pool indices: k datasets (random per draw), total // k slices each."""
    per = total // k
    eligible = [ds for ds in sorted(pool) if len(pool[ds]) >= per]
    if len(eligible) < k:
        raise SystemExit(f"only {len(eligible)} datasets have >= {per} pool slices; cannot draw {k}")
    chosen = sorted(np.random.default_rng([2000 + draw, k]).choice(eligible, k, replace=False).tolist())
    out = {}
    for ds in chosen:
        i = sorted(pool).index(ds)
        order = np.random.default_rng([3000 + draw, i]).permutation(len(pool[ds]))
        out[ds] = sorted(int(j) for j in order[:per])
    return out
