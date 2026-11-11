#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoder-only data-size and data-variety runs (frozen SAM ViT-B encoder), all in one process.

Questions
  data size     does in-domain (and zero-shot) Dice keep rising with more slices per dataset?
                arms "sam-enc_sam-dec_n<N>" (decoder trained) and "sam-enc_sam-dec-lora_n<N>" (LoRA on the
                decoder), N slices per dataset (equal for every dataset, so BraTS does not dominate), seeds.
  data variety  at a FIXED total number of slices, does training on more datasets (more structures) help
                zero-shot? arms "sam-enc_sam-dec_k<k>-draw<d>": k of the 11 datasets, total // k slices each.
                The clean datasets left out of a draw are extra unseen test sets (analyse_curve.py).
  similar data  does adding a dataset similar to an unseen one help on it? --extra-datasets MSK_T2 adds arms
                "sam-enc_sam-dec_n<N>_plus-MSK_T2": the 11 datasets' N slices (exactly those of the run
                without the extra) + N slices of each extra dataset, from <extra-root>/<dataset>/training/.
How
  The pool (data_pools.py: up to POOL_MAX slices per dataset, validation slices removed) is encoded ONCE
  with the frozen SAM encoder and kept in GPU memory (fp16, 2 MiB per slice); every run then trains a
  fresh SAM decoder on its subset exactly as train_decoder.py does (AdamW wd 0.1, Dice + 10 x Focal, one
  random structure per slice per epoch, +-10 px box jitter, best epoch on the shared validation slices).
  Runs with <= --long-threshold slices: --epochs / --patience; larger runs: --epochs-long / --patience-long.
Outputs (one folder per run, same files as train_decoder.py, so eval_decoder.py loads them):
  <out-dir>/<arm>/seed<k>/{best_decoder.pt, curve.csv, meta.json, DONE}; meta.json adds the training
  datasets, slices per dataset, train seconds, this run's share of the encoding time, peak GPU memory.
  <out-dir>/pool.json: pool sizes and the encoding time.
Resumable: finished runs (DONE) are skipped; the pool is re-encoded on restart (~minutes).

Usage (normally via run_curve_experiments.sh):
  python embedding_assess/train_decoder_curve.py \\
      --work-dir /scratch/user/s4670484/Eval_results/embedding_research_clean \\
      --val-rows /scratch/user/s4670484/Eval_results/decoder_swap/runs/val_rows.json \\
      --ckpt-dir /scratch/user/s4670484/pretrained_ckpt \\
      --out-dir /scratch/user/s4670484/Eval_results/decoder_swap/runs_curve \\
      --lr-decoder 1e-4 --lr-decoder-lora 5e-4
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
import data_pools as dp  # noqa: E402
import decoder_lib as dl  # noqa: E402
import emb_common as ec  # noqa: E402
from train_decoder import encode_images  # noqa: E402


def train_one(arm, seed, sel, pool_index, embs, packed, wd, val_inst, args, device, lr, lora, enc_s_per_slice):
    """sel: dataset -> pool indices. Trains one fresh decoder; writes the run folder."""
    run_dir = os.path.join(args.out_dir, arm, f"seed{seed}")
    if os.path.isfile(os.path.join(run_dir, "DONE")):
        print(f"{arm} seed {seed}: done - skipping")
        return
    os.makedirs(run_dir, exist_ok=True)
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx_all = [pool_index[(ds, j)] for ds in sorted(sel) for j in sel[ds]]
    keep = [i for i in idx_all if packed[i]]
    n_slices = len(idx_all)
    long_run = n_slices > args.long_threshold
    epochs, patience = (args.epochs_long, args.patience_long) if long_run else (args.epochs, args.patience)

    model = dl.build_decoder("sam", args.ckpt_dir, device, log=lambda *a: None)
    if lora:
        params = dl.add_decoder_lora(model.mask_decoder, args.rank, args.alpha)
    else:
        params = list(model.mask_decoder.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
    from utils.losses import DiceFocalLoss
    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction="mean", lambda_dice=1, lambda_focal=10)

    def validate():
        model.eval()
        d = [x[2] for x in dl.predict_instances(model, wd, val_inst, device)]
        model.train()
        return float(np.nanmean(d))

    def save():
        torch.save(dl.merged_state(model.mask_decoder) if lora else model.mask_decoder.state_dict(),
                   os.path.join(run_dir, "best_decoder.pt"))

    shape = (dl.LOWRES, dl.LOWRES)

    def batches():
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

    v0 = validate()
    best, best_ep, since = v0, 0, 0
    save()
    curve = [{"epoch": 0, "train_loss": float("nan"), "val_dice": v0, "seconds": round(time.time() - t0, 1)}]
    print(f"\n{arm} seed {seed}: {n_slices} slices from {len(sel)} datasets, lr {lr}, "
          f"{epochs} epochs max (patience {patience}); validation Dice before training {v0:.3f}", flush=True)
    model.train()
    for ep in range(1, epochs + 1):
        tl, nb, te = 0.0, 0, time.time()
        for emb, tgt, box in batches():
            loss = loss_fn(model(emb, box), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl, nb = tl + loss.item(), nb + 1
        v = validate()
        curve.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_dice": v, "seconds": round(time.time() - t0, 1)})
        if v > best + 1e-4:
            best, best_ep, since = v, ep, 0
            save()
        else:
            since += 1
        if ep == 1 or ep % 10 == 0 or since >= patience:
            print(f"  epoch {ep:3d}: train loss {tl / max(nb, 1):.4f} | val Dice {v:.4f} (best {best:.4f} at "
                  f"{best_ep}) | {time.time() - te:.1f} s/epoch", flush=True)
        if since >= patience:
            break
    with open(os.path.join(run_dir, "curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    train_s = time.time() - t0
    peak = dl.peak_gpu_gb(device)
    meta = {"arm": arm, "arm_name": dl.arm_info(arm)[2], "seed": seed, "lr": lr, "decoder_lora_rank": args.rank if lora else None,
            "batch_size": args.batch_size, "jitter": args.jitter, "datasets": sorted(sel),
            "slices_per_dataset": {ds: len(v) for ds, v in sel.items()}, "training_slices": n_slices,
            "epochs_run": curve[-1]["epoch"], "best_epoch": best_ep, "best_val_dice": best, "val_dice_before": v0,
            "train_seconds": round(train_s, 1), "encode_seconds_share": round(n_slices * enc_s_per_slice, 1),
            "seconds": round(train_s + n_slices * enc_s_per_slice, 1),
            "trainable_parameters": sum(p.numel() for p in params),
            "peak_gpu_gb_including_cache": peak,
            "peak_gpu_gb_training": None if peak is None else round(peak - embs.numel() * 2 / 1024 ** 3, 2)}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    open(os.path.join(run_dir, "DONE"), "w").close()
    print(f"  done: best val Dice {best:.4f} at epoch {best_ep} | {train_s / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--val-rows", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sizes", type=int, nargs="*", default=[15, 50, 150, 500, 1500, 5000])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--modes", nargs="*", default=["decoder", "decoder-lora"], choices=["decoder", "decoder-lora"])
    ap.add_argument("--variety-ks", type=int, nargs="*", default=[1, 3, 6, 11])
    ap.add_argument("--variety-draws", type=int, default=3)
    ap.add_argument("--variety-total", type=int, default=1500)
    ap.add_argument("--pool-max", type=int, default=dp.POOL_MAX)
    ap.add_argument("--extra-root", default=None, help="root holding <dataset>/training/ for --extra-datasets")
    ap.add_argument("--extra-datasets", nargs="*", default=[], help="datasets added on top of the 11 (plus arms)")
    ap.add_argument("--lr-decoder", type=float, default=1e-4)
    ap.add_argument("--lr-decoder-lora", type=float, default=5e-4)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--jitter", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--epochs-long", type=int, default=30)
    ap.add_argument("--patience-long", type=int, default=5)
    ap.add_argument("--long-threshold", type=int, default=2000, help="runs with more slices use --epochs-long")
    ap.add_argument("--encode-batch", type=int, default=8, help="slices per encoder pass while building the pool")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    workers = args.workers if args.workers is not None else max(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1, 0)
    print(sys.executable, "| torch", torch.__version__, "| hip", torch.version.hip)
    if args.device != "cpu":
        assert torch.cuda.is_available(), "no GPU visible - wrong conda env for this card?"
        print("device:", torch.cuda.get_device_name(0))
    device = torch.device(args.device)

    wd = dl.WorkDir(args.work_dir, "sam_vit_b__sam")
    val_rows = set(json.load(open(args.val_rows)))
    val_inst = wd.instances(sorted(val_rows))
    pool = dp.training_pool(wd.manifest, val_rows, args.pool_max)
    pairs, pool_index = [], {}
    for ds in sorted(pool):
        for j, p in enumerate(pool[ds]):
            pool_index[(ds, j)] = len(pairs)
            pairs.append(p)
    extra = dp.extra_pool(args.extra_root, args.extra_datasets, args.pool_max) if args.extra_datasets else {}
    clash = set(extra) & set(pool)
    if clash:
        raise SystemExit(f"extra datasets already among the 11: {sorted(clash)}")
    for ds in sorted(extra):
        for j, p in enumerate(extra[ds]):
            pool_index[(ds, j)] = len(pairs)
            pairs.append(p)
    for ds in sorted(pool):
        print(f"  pool {ds}: {len(pool[ds])} slices")
    for ds in sorted(extra):
        print(f"  pool {ds} (extra, from {args.extra_root}): {len(extra[ds])} slices")
    print(f"pool: {len(pairs)} slices ({len(pairs) * 2 / 1024:.1f} GB on GPU once encoded)", flush=True)

    _, encoder = ec.load_model("sam_vit_b", args.ckpt_dir, device)
    encoder.eval()
    chk = sorted(val_rows)[:4]
    with torch.no_grad():
        cur = [encoder(ec.preprocess(ec.load_slice(wd.manifest[r]["img_path"]), "sam")[0][None].to(device))[0]
               .float().cpu().numpy() for r in chk]
    cos = [float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
           for a, b in zip(cur, (wd.emb(r).astype(np.float32) for r in chk))]
    print(f"[check] encoder output vs stored features ({len(chk)} slices): cosine min {min(cos):.5f}"
          + ("" if min(cos) > 0.999 else "  <-- MISMATCH: preprocessing differs, stop and check"), flush=True)
    te = time.time()
    embs, packed = encode_images(pairs, encoder, device, workers, batch_size=args.encode_batch)
    enc_s = time.time() - te
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"encoded {len(pairs)} slices in {enc_s / 60:.1f} min ({enc_s / max(len(pairs), 1) * 1000:.0f} ms per slice)",
          flush=True)
    with open(os.path.join(args.out_dir, "pool.json"), "w") as f:
        json.dump({"pool_max": args.pool_max, "slices_per_dataset": {d: len(v) for d, v in {**pool, **extra}.items()},
                   "encode_seconds": round(enc_s, 1), "encode_seconds_per_slice": enc_s / max(len(pairs), 1)}, f, indent=1)
    per_slice = enc_s / max(len(pairs), 1)

    # ---- data size ---------------------------------------------------------------------
    for mode in args.modes:
        lora = mode == "decoder-lora"
        for n in args.sizes:
            for seed in args.seeds:
                arm = f"sam-enc_sam-dec{'-lora' if lora else ''}_n{n}"
                train_one(arm, seed, dp.size_subset(pool, n, seed), pool_index, embs, packed, wd, val_inst, args,
                          device, args.lr_decoder_lora if lora else args.lr_decoder, lora, per_slice)
    # ---- similar data: the 11 datasets' slices + the extra datasets -------------------------
    if extra:
        tag = "+".join(sorted(extra))
        for mode in args.modes:
            lora = mode == "decoder-lora"
            for n in args.sizes:
                for seed in args.seeds:
                    sel = dp.size_subset(pool, n, seed)
                    sel.update(dp.extra_subset(extra, n, seed))
                    train_one(f"sam-enc_sam-dec{'-lora' if lora else ''}_n{n}_plus-{tag}", seed, sel, pool_index,
                              embs, packed, wd, val_inst, args, device,
                              args.lr_decoder_lora if lora else args.lr_decoder, lora, per_slice)
    # ---- data variety (decoder trained, fixed total) -------------------------------------
    for k in args.variety_ks:
        for draw in range(args.variety_draws):
            sel = dp.variety_subset(pool, k, args.variety_total, draw)
            train_one(f"sam-enc_sam-dec_k{k}-draw{draw}", 0, sel, pool_index, embs, packed, wd, val_inst, args,
                      device, args.lr_decoder, False, per_slice)
    print(f"\nall runs finished in {(time.time() - t0) / 3600:.1f} h")


if __name__ == "__main__":
    main()
