#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Which SAMRI checkpoint produced the published SAMRI numbers? (CPU, ~1 min)

The swapped-encoder evaluation reproduced the published SAM ViT-B and MedSAM Dice
instance by instance (correlation 0.999 / 0.993), but NOT the published SAMRI Dice with
pretrained_ckpt/samri_vitb_box.pth and samri_vitb_bp.pth (correlation 0.75 / 0.46,
mean of dataset medians 0.838 / 0.769 vs published 0.887 / 0.891). This script compares
every given checkpoint with SAM ViT-B, part by part, and with each other.

Expected for a genuine SAMRI checkpoint (model.py freezes the image and prompt encoders):
image_encoder and prompt_encoder identical to SAM (max |diff| 0), mask_decoder different.

Usage (repo root, any env with torch):
  python embedding_assess/ckpt_check.py \
      --sam /scratch/user/s4670484/pretrained_ckpt/sam_vit_b_01ec64.pth \
      --ckpts /scratch/user/s4670484/pretrained_ckpt/samri_vitb_*.pth \
              /scratch/user/s4670484/comparison_ckpt_samri/*.pth
"""

import argparse
import os
import time

import torch


def load(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return {k.replace("module.", "", 1): v for k, v in sd.items() if torch.is_tensor(v)}


def part_diff(a, b, prefix):
    ka = {k for k in a if k.startswith(prefix)}
    kb = {k for k in b if k.startswith(prefix)}
    common = ka & kb
    if not common:
        return f"{prefix:15s} missing (keys: {len(ka)} vs {len(kb)})"
    same = [k for k in common if a[k].shape == b[k].shape]
    mism = sorted(set(common) - set(same))
    extra = "" if ka == kb else f", key sets differ ({len(ka)} vs {len(kb)})"
    if mism:  # e.g. a ViT-H or ViT-L checkpoint compared with ViT-B
        k = mism[0]
        extra += (f", {len(mism)}/{len(common)} tensors differ in shape "
                  f"(e.g. {k}: {tuple(a[k].shape)} vs {tuple(b[k].shape)})")
    if not same:
        return f"{prefix:15s} not comparable{extra}"
    d = max((a[k].float() - b[k].float()).abs().max().item() for k in same)
    return f"{prefix:15s} max |diff| {d:.3g}{extra}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    args = ap.parse_args()
    sam = load(args.sam)
    parts = ("image_encoder", "prompt_encoder", "mask_decoder")
    decoders = {}
    for p in args.ckpts:
        if not os.path.isfile(p):
            print(f"\n{p}: not found")
            continue
        st = os.stat(p)
        print(f"\n{p}\n  {st.st_size / 1e6:.0f} MB, modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))}")
        try:
            c = load(p)
        except Exception as e:  # noqa: BLE001
            print(f"  could not load: {e}")
            continue
        print(f"  {len(c)} tensors; vs SAM ViT-B:")
        for part in parts:
            print("   ", part_diff(c, sam, part))
        decoders[os.path.basename(p)] = {k: v for k, v in c.items() if k.startswith("mask_decoder")}
    names = list(decoders)
    if len(names) > 1:
        print("\nmask decoders, pairwise max |diff| (0 = same weights):")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                print(f"  {a}  vs  {b}: {part_diff(decoders[a], decoders[b], 'mask_decoder').split('max |diff| ')[-1]}")


if __name__ == "__main__":
    main()
