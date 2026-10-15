#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ONE SAM mask decoder (one arm, one seed) on frozen encoder output, with
SAMRI's recipe (see decoder_lib.py). Used for the swapped-encoder experiment.

Data sources
  --source manifest  the manifest's training rows (minus the held-out validation
                     rows), using the stored encoder output in
                     <work-dir>/features/<arm features>/neck.npy (the 'small data'
                     arms: 11 clean datasets x 150 slices)
  --source images    every training slice of the manifest's datasets, read from
                     the same training/ folders the manifest's training rows come
                     from (the 'all11' data-scale arm; SAM encoder only). The frozen
                     SAM ViT-B encoder is run once at the start and its output is
                     kept in GPU memory only - nothing is written to disk. If the
                     slices would exceed --max-gpu-gb, every dataset is capped at
                     the same number of slices (smaller datasets keep all of theirs).
                     Validation slices are removed by full path.
Validation (every epoch, same rows for every arm): tight box per structure on the
held-out training rows, Dice on native pixels; the best epoch is kept.

Outputs in <out-dir>/<arm>/seed<k>/: best_decoder.pt (mask decoder weights),
curve.csv (epoch, train loss, validation Dice), meta.json, DONE.

Usage (normally via run_decoder_experiments.sh):
  python embedding_assess/train_decoder.py --arm sam-enc_sam-dec_small --seed 0 \
      --work-dir /scratch/user/s4670484/Eval_results/embedding_research_clean \
      --ckpt-dir /scratch/user/s4670484/pretrained_ckpt \
      --out-dir /scratch/user/s4670484/Eval_results/decoder_swap/runs --lr 1e-4
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_lib as dl  # noqa: E402
import emb_common as ec  # noqa: E402


EMB_BYTES = 256 * ec.GRID * ec.GRID * 2  # one fp16 neck output: 2 MiB


class ImageSet(torch.utils.data.Dataset):
    """Training slices from image files: SAM-preprocessed input + packed 256x256 targets
    (one per label, SAM pad frame) - targets are bit-packed to keep host memory small."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        raw, seg = ec.load_slice(*self.pairs[k])
        x, _ = ec.preprocess(raw, "sam")
        ts = [dl.lowres_target(seg == lab, "sam") for lab in ec.slice_labels(seg)]
        return x, k, [np.packbits(t) for t in ts if t.any()]


def capped_sample(by_ds, max_n, seed):
    """Equal cap per dataset so the total fits max_n; datasets below the cap keep all."""
    counts = {d: len(v) for d, v in by_ds.items()}
    if sum(counts.values()) <= max_n:
        return by_ds, None
    lo, hi = 0, max(counts.values())
    while lo < hi:  # largest cap L with sum(min(c, L)) <= max_n
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if sum(min(c, mid) for c in counts.values()) <= max_n else (lo, mid - 1)
    rng = np.random.default_rng(seed)
    return {d: (v if len(v) <= lo else sorted(v[i] for i in rng.choice(len(v), lo, replace=False)))
            for d, v in by_ds.items()}, lo


def as_list(batch):  # top level so DataLoader workers can pickle it
    return batch


def encode_images(pairs, encoder, device, workers, batch_size=8):
    """Runs the frozen encoder over every pair; returns fp16 GPU tensor + packed targets."""
    embs = torch.empty((len(pairs), 256, ec.GRID, ec.GRID), dtype=torch.float16, device=device)
    tgts = [None] * len(pairs)
    loader = torch.utils.data.DataLoader(ImageSet(pairs), batch_size=batch_size, num_workers=workers,
                                         shuffle=False, collate_fn=as_list)
    t1, n = time.time(), len(loader)
    for b, items in enumerate(loader):
        x = torch.stack([it[0] for it in items]).to(device)
        with torch.no_grad():
            neck = encoder(x).half()
        for j, (_, k, ts) in enumerate(items):
            embs[k] = neck[j]
            tgts[k] = ts
        if b == 0 or (b + 1) % 500 == 0 or b + 1 == n:
            el = time.time() - t1
            print(f"  encoding batch {b + 1}/{n}  {el:.0f} s elapsed, ~{el / (b + 1) * (n - b - 1):.0f} s left",
                  flush=True)
    return embs, tgts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(dl.ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", required=True, help="in-domain work dir (manifest, geometry, features)")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default=None, choices=["manifest", "images"],
                    help="default: 'images' for the all11 arm, else 'manifest'")
    ap.add_argument("--max-gpu-gb", type=float, default=100,
                    help="GPU memory for the encoder output (--source images); 2 MiB per slice")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)  # SAMRI
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15, help="stop after this many epochs without a better val Dice")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--jitter", type=int, default=10)  # SAMRI cfg.JITTER
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="", help="optional suffix for the run folder (e.g. the LR pilot)")
    args = ap.parse_args()
    t0 = time.time()

    feats, start, arm_name = dl.ARMS[args.arm]
    source = args.source or ("images" if args.arm.endswith("all11") else "manifest")
    run_dir = os.path.join(args.out_dir, args.arm + args.tag, f"seed{args.seed}")
    if os.path.isfile(os.path.join(run_dir, "DONE")):
        print(f"{run_dir}: already finished - skipping")
        return
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    workers = args.workers if args.workers is not None else max(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1, 0)

    print(sys.executable, "| torch", torch.__version__, "| hip", torch.version.hip)
    if args.device != "cpu":
        assert torch.cuda.is_available(), "no GPU visible - wrong conda env for this card?"
        print("device:", torch.cuda.get_device_name(0))
    device = torch.device(args.device)
    print(f"arm: {arm_name} | features {feats} | decoder start {start} | source {source} | "
          f"seed {args.seed} | lr {args.lr} | batch {args.batch_size}")

    wd = dl.WorkDir(args.work_dir, feats)
    val_rows = set(dl.val_split(wd.manifest, args.val_frac, args.split_seed,
                                os.path.join(args.out_dir, "val_rows.json")))
    val_inst = wd.instances(sorted(val_rows))
    model = dl.build_decoder(start, args.ckpt_dir, device)
    opt = torch.optim.AdamW(model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from utils.losses import DiceFocalLoss
    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction="mean", lambda_dice=1, lambda_focal=10)

    def validate():
        model.eval()
        d = [x[2] for x in dl.predict_instances(model, wd, val_inst, device)]
        model.train()
        return float(np.nanmean(d))

    v0 = validate()
    print(f"[sanity] decoder before training, validation Dice {v0:.3f} over {len(val_inst)} structures "
          f"in {len(val_rows)} held-out slices", flush=True)

    # ---- training data -----------------------------------------------------------
    cap_info = {}
    if source == "manifest":
        train_rows = [r for r in wd.rows("train") if r not in val_rows]
        embs = torch.stack([torch.as_tensor(wd.emb(r)) for r in train_rows]).to(device)  # fp16 on GPU
        inst = wd.instances(train_rows)
        tgt_by_row = {}
        for r, lab, m in inst:
            t = dl.lowres_target(m, wd.pipeline)
            if t.any():
                tgt_by_row.setdefault(r, []).append(t)
        keep = [i for i, r in enumerate(train_rows) if r in tgt_by_row]
        print(f"training: {len(keep)} slices, {sum(len(v) for v in tgt_by_row.values())} structures "
              f"({time.time() - t0:.0f} s to load)", flush=True)

        def batches(epoch):
            order = rng.permutation(keep)
            for s in range(0, len(order), args.batch_size):
                idx = order[s:s + args.batch_size]
                ts, bs = [], []
                for i in idx:
                    opts = tgt_by_row[train_rows[i]]
                    t = opts[int(rng.integers(len(opts)))]  # one random label per slice per epoch (SAMRI)
                    ts.append(t)
                    bs.append(dl.jitter_box_1024(dl.tight_box_256(t), args.jitter, rng))
                yield (embs[torch.as_tensor(idx, device=device)].float(),
                       torch.as_tensor(np.stack(ts), dtype=torch.float32, device=device)[:, None],
                       torch.as_tensor(np.stack(bs), device=device))
        n_batches = (len(keep) + args.batch_size - 1) // args.batch_size
    else:
        if wd.pipeline != "sam":
            sys.exit("--source images supports the SAM encoder (sam pipeline) only")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from build_manifest import list_pairs
        val_paths = {wd.manifest[r]["img_path"] for r in val_rows}
        folders = {}
        for r in wd.manifest:
            if r["split"] == "train":
                folders[r["dataset"]] = os.path.dirname(r["img_path"])
        by_ds, seen_val = {}, 0
        for ds in sorted(folders):
            pairs = list_pairs(folders[ds])[0]
            seen_val += sum(p[0] in val_paths for p in pairs)
            by_ds[ds] = [p for p in pairs if p[0] not in val_paths]
        max_n = int(args.max_gpu_gb * 1024 ** 3 // EMB_BYTES)
        by_ds, cap = capped_sample(by_ds, max_n, args.split_seed)
        for ds in sorted(by_ds):
            print(f"  {ds}: {len(by_ds[ds])} training slices from {folders[ds]}")
        if seen_val != len(val_paths):
            sys.exit(f"only {seen_val} of {len(val_paths)} validation slices found in the training folders - stopping "
                     "(validation would overlap training)")
        print(f"training: {sum(map(len, by_ds.values()))} slices ({len(val_paths)} validation slices removed)"
              + (f"; CAPPED at {cap} slices per dataset to fit --max-gpu-gb {args.max_gpu_gb}" if cap else "; no cap"),
              flush=True)
        cap_info = {"slices_per_dataset": {d: len(v) for d, v in by_ds.items()}, "cap_per_dataset": cap}
        pairs = [p for ds in sorted(by_ds) for p in by_ds[ds]]
        _, encoder = ec.load_model("sam_vit_b", args.ckpt_dir, device)
        encoder.eval()
        # check: on-the-fly encoder output must equal the stored features the validation uses
        chk = sorted(val_rows)[:4]
        with torch.no_grad():
            cur = encoder(torch.stack([ec.preprocess(ec.load_slice(wd.manifest[r]["img_path"]), "sam")[0]
                                       for r in chk]).to(device)).float().cpu().numpy()
        cos = [float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
               for a, b in zip(cur, (wd.emb(r).astype(np.float32) for r in chk))]
        print(f"[check] encoder output now vs stored features ({len(chk)} slices): cosine min {min(cos):.5f}"
              + ("" if min(cos) > 0.999 else "  <-- MISMATCH: preprocessing differs, stop and check"), flush=True)
        embs, packed = encode_images(pairs, encoder, device, workers)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
        keep = [i for i, ts in enumerate(packed) if ts]
        print(f"encoded {len(pairs)} slices ({embs.numel() * 2 / 1024 ** 3:.1f} GB on GPU), {len(keep)} with "
              f"structures, {sum(len(ts) for ts in packed)} structures ({time.time() - t0:.0f} s)", flush=True)
        shape = (dl.LOWRES, dl.LOWRES)

        def batches(epoch):
            order = rng.permutation(keep)
            for s in range(0, len(order), args.batch_size):
                idx = order[s:s + args.batch_size]
                ts, bs = [], []
                for i in idx:
                    opts = packed[i]
                    t = np.unpackbits(opts[int(rng.integers(len(opts)))])[:shape[0] * shape[1]].reshape(shape)
                    ts.append(t)
                    bs.append(dl.jitter_box_1024(dl.tight_box_256(t), args.jitter, rng))
                yield (embs[torch.as_tensor(idx, device=device)].float(),
                       torch.as_tensor(np.stack(ts), dtype=torch.float32, device=device)[:, None],
                       torch.as_tensor(np.stack(bs), device=device))
        n_batches = (len(keep) + args.batch_size - 1) // args.batch_size

    # ---- train ---------------------------------------------------------------------
    best, best_ep, since = v0, 0, 0
    torch.save(model.mask_decoder.state_dict(), os.path.join(run_dir, "best_decoder.pt"))
    curve = [{"epoch": 0, "train_loss": float("nan"), "val_dice": v0, "seconds": round(time.time() - t0, 1)}]
    model.train()
    for ep in range(1, args.epochs + 1):
        tl, nb, te = 0.0, 0, time.time()
        for b, (emb, tgt, box) in enumerate(batches(ep)):
            loss = loss_fn(model(emb, box), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl, nb = tl + loss.item(), nb + 1
            if ep == 1 and b in (0, 9):
                print(f"  epoch 1 batch {b + 1}/{n_batches}: loss {loss.item():.4f} ({time.time() - te:.1f} s)",
                      flush=True)
        v = validate()
        curve.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_dice": v, "seconds": round(time.time() - t0, 1)})
        improved = v > best + 1e-4
        if improved:
            best, best_ep, since = v, ep, 0
            torch.save(model.mask_decoder.state_dict(), os.path.join(run_dir, "best_decoder.pt"))
        else:
            since += 1
        print(f"epoch {ep:3d}: train loss {tl / max(nb, 1):.4f} | val Dice {v:.4f}"
              f"{'  *best' if improved else ''} | {time.time() - te:.0f} s", flush=True)
        with open(os.path.join(run_dir, "curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
            w.writeheader()
            w.writerows(curve)
        if since >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    meta = {"arm": args.arm, "arm_name": arm_name, "features": feats, "decoder_start": start, "source": source,
            "seed": args.seed, "lr": args.lr, "batch_size": args.batch_size, "jitter": args.jitter,
            "epochs_run": curve[-1]["epoch"], "best_epoch": best_ep, "best_val_dice": best, "val_dice_before": v0,
            "seconds": round(time.time() - t0, 1), "tag": args.tag, **cap_info}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    open(os.path.join(run_dir, "DONE"), "w").close()
    print(f"done: best val Dice {best:.4f} at epoch {best_ep} (before training {v0:.4f}) -> {run_dir}")


if __name__ == "__main__":
    main()
