#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train ONE image-trained arm (SAM ViT-B; encoder + decoder LoRA, or full fine-tuning), then evaluate it.

Question: is cheap encoder tuning (LoRA) worth it? Compare with LoRA on the decoder alone (frozen
encoder, train_decoder.py arms sam-enc_sam-dec-lora_*), with the fully trained decoder on the frozen
encoder, and with MedSAM's fully fine-tuned encoder.

Model
  SAM ViT-B with every original weight frozen. LoRA on the query and value projections of every
  encoder block (as in SAMed) and of every attention layer of the mask decoder's two-way transformer:
  W x + (alpha / r) * B A x, with B = 0 at the start, so the model starts EXACTLY as SAM (the encoder
  is checked at start against the stored features). SAM's prompt encoder is frozen and shared.
  (A "full" mode - every encoder and decoder weight trains - is implemented but no arm uses it.)
Recipe (same as train_decoder.py / SAMRI): AdamW, weight decay 0.1, Dice + 10 x Focal on the 256x256
  low-res frame, one random structure per slice per epoch, box from that frame with +-10 px jitter,
  validation Dice on the same held-out slices (decoder_swap/runs/val_rows.json), best epoch kept.
  bf16 autocast; the encoder runs on every step, so images are read and preprocessed each epoch.
Data
  small   the in-domain manifest's training rows minus the validation rows (1,485 slices)
  all11   every training slice of the same 11 datasets, capped exactly as the decoder all-slices arm
          (same seed and limit -> the same 51,198 slices); --samples-per-epoch bounds an epoch.
  n<N>    N slices per dataset from the shared pool (data_pools.py); with the same --seed these are exactly
          the slices of the decoder-only data-size run sam-enc_sam-dec_n<N> (train_decoder_curve.py).
Cost: meta.json records train_seconds (training only, without the final test), trainable parameters and
  peak GPU memory, for the compute comparison (analyse_curve.py).
Evaluation (end of training, best weights): the encoder must be rerun on the test images (its
  output differs from the stored features), so this script scores itself on the in-domain test rows
  and the zero-shot test rows, same boxes and Dice as eval_decoder.py, and writes eval_instances.csv
  in eval_decoder.py's format (merge with: eval_decoder.py --extra-instances ...).
Resumable: last.pt is written every epoch and picked up on resubmission.
Evaluate-only: --eval-only LABEL=DIR reloads <run>/best.pt (no training) and scores it on DIR's test rows
  (e.g. knee=<work dir of MSK_PD/MSK_FLASH/MSK_T2>), writing <run>/eval_instances_<LABEL>.csv for
  eval_decoder.py --extra-instances.

Outputs in <out-dir>/<arm><tag>/seed<k>/: best.pt (encoder LoRA + decoder state), best_decoder.pt (decoder
with its LoRA merged), curve.csv, meta.json, eval_instances.csv, DONE.

Usage (normally via run_lora_experiments.sh):
  python embedding_assess/train_lora.py --arm sam-enc-lora_sam-dec-lora_small --seed 0 --lr 1e-4 \\
      --work-dir /scratch/user/s4670484/Eval_results/embedding_research_clean \\
      --zeroshot-dir /scratch/user/s4670484/Eval_results/embedding_research_zeroshot \\
      --val-rows /scratch/user/s4670484/Eval_results/decoder_swap/runs/val_rows.json \\
      --ckpt-dir /scratch/user/s4670484/pretrained_ckpt \\
      --out-dir /scratch/user/s4670484/Eval_results/decoder_swap/runs_lora
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decoder_lib as dl  # noqa: E402
import emb_common as ec  # noqa: E402

DECODER_ALL11_MAX = int(100 * 1024 ** 3 // (256 * ec.GRID * ec.GRID * 2))  # train_decoder.py --max-gpu-gb 100


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRAQKV(nn.Module):
    """Wraps a frozen qkv nn.Linear(dim, 3*dim); adds low-rank updates to q and v."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.dim, self.scale = base, base.in_features, alpha / r
        # same device/dtype as the wrapped layer: the adapters are added after the encoder is on the GPU
        kw = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a_q = nn.Parameter(torch.empty(r, self.dim, **kw))
        self.lora_b_q = nn.Parameter(torch.zeros(self.dim, r, **kw))
        self.lora_a_v = nn.Parameter(torch.empty(r, self.dim, **kw))
        self.lora_b_v = nn.Parameter(torch.zeros(self.dim, r, **kw))
        nn.init.kaiming_uniform_(self.lora_a_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_a_v, a=math.sqrt(5))

    def forward(self, x):
        out = self.base(x)
        q = (x @ self.lora_a_q.t()) @ self.lora_b_q.t()
        v = (x @ self.lora_a_v.t()) @ self.lora_b_v.t()
        return out + self.scale * torch.cat([q, torch.zeros_like(q), v], dim=-1)


class CheckpointBlock(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.blk = blk

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self.blk, x, use_reentrant=False)


def add_lora(encoder, r, alpha, grad_checkpoint):
    for p in encoder.parameters():
        p.requires_grad = False
    for blk in encoder.blocks:
        blk.attn.qkv = LoRAQKV(blk.attn.qkv, r, alpha)
    if grad_checkpoint:
        encoder.blocks = nn.ModuleList([CheckpointBlock(b) for b in encoder.blocks])
    return [p for n, p in encoder.named_parameters() if "lora_" in n]


def lora_state(encoder):
    """The encoder's TRAINABLE weights (LoRA matrices, or every weight when fully fine-tuned)."""
    return {n.replace(".blk.", "."): p.detach().cpu().clone() for n, p in encoder.named_parameters() if p.requires_grad}


def load_lora_state(encoder, sd):
    params = {n.replace(".blk.", "."): p for n, p in encoder.named_parameters() if p.requires_grad}
    assert set(params) == set(sd), "trainable encoder parameter names differ from the saved state"
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(sd[n])


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class TrainSet(torch.utils.data.Dataset):
    """(image path, mask path) -> SAM-preprocessed input, one random structure's 256 target, jittered box."""

    def __init__(self, pairs, jitter, seed):
        self.pairs, self.jitter, self.seed, self.epoch = pairs, jitter, seed, 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        rng = np.random.default_rng([self.seed, self.epoch, k])
        raw, seg = ec.load_slice(*self.pairs[k])
        labs = list(ec.slice_labels(seg))
        rng.shuffle(labs)
        for lab in labs:  # first structure (in random order) that survives the 256 resize
            t = dl.lowres_target(seg == lab, "sam")
            if t.any():
                x, _ = ec.preprocess(raw, "sam")
                box = dl.jitter_box_1024(dl.tight_box_256(t), self.jitter, rng)
                return x, torch.from_numpy(t), torch.from_numpy(box)
        return None


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return tuple(torch.stack(z) for z in zip(*batch))


class LiveWorkDir(dl.WorkDir):
    """A work dir whose embeddings come from the (LoRA) encoder instead of the stored features."""

    def __init__(self, work_dir, encoder, device, autocast):
        super().__init__(work_dir, "sam_vit_b__sam")
        self.encoder, self.device, self.autocast = encoder, device, autocast

    def emb(self, r):
        x, _ = ec.preprocess(ec.load_slice(self.manifest[r]["img_path"]), "sam")
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            return self.encoder(x[None].to(self.device))[0].float().cpu().numpy()


def training_pairs(arm_data, wd, val_rows, max_slices, split_seed, seed=0):
    if arm_data == "small":
        rows = [r for r in wd.rows("train") if r not in val_rows]
        return [(wd.manifest[r]["img_path"], wd.manifest[r]["seg_path"]) for r in rows], None
    if arm_data.startswith("n"):  # N per dataset, the same slices as the decoder-only data-size run
        import data_pools as dp
        pool = dp.training_pool(wd.manifest, val_rows)
        sel = dp.size_subset(pool, int(arm_data[1:]), seed)
        for ds in sorted(sel):
            print(f"  {ds}: {len(sel[ds])} training slices")
        return [pool[ds][j] for ds in sorted(sel) for j in sel[ds]], None
    from build_manifest import list_pairs
    from train_decoder import capped_sample
    val_paths = {wd.manifest[r]["img_path"] for r in val_rows}
    folders = {r["dataset"]: os.path.dirname(r["img_path"]) for r in wd.manifest if r["split"] == "train"}
    by_ds, seen = {}, 0
    for ds in sorted(folders):
        pairs = list_pairs(folders[ds])[0]
        seen += sum(p[0] in val_paths for p in pairs)
        by_ds[ds] = [p for p in pairs if p[0] not in val_paths]
    if seen != len(val_paths):
        sys.exit(f"only {seen} of {len(val_paths)} validation slices found in the training folders - stopping")
    by_ds, cap = capped_sample(by_ds, max_slices, split_seed)
    for ds in sorted(by_ds):
        print(f"  {ds}: {len(by_ds[ds])} training slices")
    return [p for ds in sorted(by_ds) for p in by_ds[ds]], cap


# ---------------------------------------------------------------------------
def eval_only(args):
    label, _, wdir = args.eval_only.partition("=")
    if not label or not wdir:
        sys.exit(f"--eval-only expects LABEL=DIR, got {args.eval_only!r}")
    arm_data, mode, arm_name = dl.IMAGE_ARMS[args.arm]
    run_dir = os.path.join(args.out_dir, args.arm + args.tag, f"seed{args.seed}")
    out = os.path.join(run_dir, f"eval_instances_{label}.csv")
    if os.path.isfile(out):
        print(f"{out}: exists - skipping")
        return
    best = os.path.join(run_dir, "best.pt")
    if not os.path.isfile(best):
        sys.exit(f"no trained weights: {best}")
    meta_p = os.path.join(run_dir, "meta.json")
    meta = json.load(open(meta_p)) if os.path.isfile(meta_p) else {}
    rank, alpha = meta.get("rank", args.rank), meta.get("alpha", args.alpha)
    t0 = time.time()
    if args.device != "cpu":
        assert torch.cuda.is_available(), "no GPU visible - wrong conda env for this card?"
        print("device:", torch.cuda.get_device_name(0))
    device = torch.device(args.device)
    autocast = device.type == "cuda" and not args.no_autocast
    _, encoder = ec.load_model("sam_vit_b", args.ckpt_dir, device)
    model = dl.build_decoder("sam", args.ckpt_dir, device, log=lambda *a: None)
    if mode == "lora":
        add_lora(encoder, rank, alpha, False)
        dl.add_decoder_lora(model.mask_decoder, rank, alpha)
    else:
        for p in encoder.parameters():
            p.requires_grad = True
    st = torch.load(best, map_location="cpu")
    load_lora_state(encoder, st["lora"])
    model.mask_decoder.load_state_dict(st["decoder"])
    encoder.eval(), model.eval()
    lw = LiveWorkDir(wdir, encoder, device, autocast)
    recs = []
    for r, lab, d, area in dl.predict_instances(model, lw, lw.instances(lw.rows("test")), device):
        m = lw.manifest[r]
        recs.append({"system": args.arm + args.tag, "name": arm_name + (f" ({args.tag.strip('_')})" if args.tag else ""),
                     "seed": args.seed, "test_set": label, "root": m["root"], "dataset": m["dataset"],
                     "row": r, "label": lab, "area": area, "dice": d})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    print(f"{arm_name} on {label}: median Dice {np.nanmedian([x['dice'] for x in recs]):.3f} over {len(recs)} "
          f"structures ({time.time() - t0:.0f} s) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(dl.IMAGE_ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", required=True, help="in-domain work dir (manifest, stored SAM features)")
    ap.add_argument("--zeroshot-dir", default=None, help="zero-shot work dir (manifest, stored SAM features)")
    ap.add_argument("--val-rows", required=True, help="decoder_swap/runs/val_rows.json (shared validation slices)")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-4, help="for LoRA and decoder")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = every training slice each epoch")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--jitter", type=int, default=10)
    ap.add_argument("--max-slices", type=int, default=DECODER_ALL11_MAX,
                    help="all11 cap; the default reproduces the decoder all-slices arm's slice set")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--grad-checkpoint", action="store_true", help="lower memory, ~30%% slower")
    ap.add_argument("--no-autocast", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-only", default=None, metavar="LABEL=DIR",
                    help="no training: test the finished run's best.pt on DIR's test rows under the label LABEL")
    args = ap.parse_args()
    if args.eval_only:
        return eval_only(args)
    t0 = time.time()

    arm_data, mode, arm_name = dl.IMAGE_ARMS[args.arm]
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
    autocast = device.type == "cuda" and not args.no_autocast
    print(f"arm: {arm_name} | " + (f"LoRA rank {args.rank}, alpha {args.alpha}" if mode == "lora" else "full fine-tuning")
          + f" | lr {args.lr} | batch {args.batch_size} | seed {args.seed} | bf16 {autocast}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- model ---------------------------------------------------------------------
    _, encoder = ec.load_model("sam_vit_b", args.ckpt_dir, device)
    model = dl.build_decoder("sam", args.ckpt_dir, device)
    wd = dl.WorkDir(args.work_dir, "sam_vit_b__sam")
    val_rows = set(json.load(open(args.val_rows)))
    chk = sorted(val_rows)[:4]
    with torch.no_grad():  # before LoRA (B = 0 makes it identical anyway): fp32 output vs stored features
        cur = [encoder(ec.preprocess(ec.load_slice(wd.manifest[r]["img_path"]), "sam")[0][None].to(device))[0]
               .float().cpu().numpy() for r in chk]
    cos = [float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
           for a, b in zip(cur, (wd.emb(r).astype(np.float32) for r in chk))]
    print(f"[check] encoder output vs stored features ({len(chk)} slices): cosine min {min(cos):.5f}"
          + ("" if min(cos) > 0.999 else "  <-- MISMATCH: preprocessing differs, stop and check"), flush=True)
    if mode == "lora":
        lora_params = add_lora(encoder, args.rank, args.alpha, args.grad_checkpoint)
        dec_params = dl.add_decoder_lora(model.mask_decoder, args.rank, args.alpha)
    else:  # full fine-tuning: every encoder and decoder weight (prompt encoder stays frozen)
        for p in encoder.parameters():
            p.requires_grad = True
        if args.grad_checkpoint:
            encoder.blocks = nn.ModuleList([CheckpointBlock(b) for b in encoder.blocks])
        lora_params = list(encoder.parameters())
        dec_params = list(model.mask_decoder.parameters())
    encoder.train()
    n_lora = sum(p.numel() for p in lora_params) + sum(p.numel() for p in dec_params)
    print(f"trainable: encoder {sum(p.numel() for p in lora_params) / 1e6:.3f} M + decoder "
          f"{sum(p.numel() for p in dec_params) / 1e6:.3f} M parameters"
          + (" (LoRA only; all original weights frozen)" if mode == "lora" else " (everything)"), flush=True)
    opt = torch.optim.AdamW(lora_params + dec_params, lr=args.lr, weight_decay=args.weight_decay)
    from utils.losses import DiceFocalLoss
    loss_fn = DiceFocalLoss(sigmoid=True, squared_pred=True, reduction="mean", lambda_dice=1, lambda_focal=10)

    live = LiveWorkDir(args.work_dir, encoder, device, autocast)
    val_inst = live.instances(sorted(val_rows))

    def validate():
        encoder.eval(), model.eval()
        d = [x[2] for x in dl.predict_instances(model, live, val_inst, device)]
        encoder.train(), model.train()
        return float(np.nanmean(d))

    # ---- data ----------------------------------------------------------------------
    pairs, cap = training_pairs(arm_data, wd, val_rows, args.max_slices, args.split_seed, args.seed)
    tset = TrainSet(pairs, args.jitter, args.seed)
    per_epoch = args.samples_per_epoch or len(pairs)
    print(f"training: {len(pairs)} slices" + (f" (capped at {cap} per dataset)" if cap else "")
          + f"; {per_epoch} samples per epoch", flush=True)

    # ---- resume or start -------------------------------------------------------------
    last = os.path.join(run_dir, "last.pt")
    elapsed0 = 0.0  # training time of earlier, cut submissions of this run
    if os.path.isfile(last):
        st = torch.load(last, map_location="cpu")
        load_lora_state(encoder, st["lora"])
        model.mask_decoder.load_state_dict(st["decoder"])
        opt.load_state_dict(st["opt"])
        start, best, best_ep, since, curve = st["epoch"] + 1, st["best"], st["best_ep"], st["since"], st["curve"]
        rng = np.random.default_rng([args.seed, start])
        elapsed0 = st.get("elapsed", 0.0)
        print(f"resumed after epoch {st['epoch']} (best val Dice {best:.4f} at epoch {best_ep})", flush=True)
    else:
        v0 = validate()
        print(f"[sanity] before training, validation Dice {v0:.3f} over {len(val_inst)} structures "
              f"(the frozen-encoder SAM arm started at 0.795)", flush=True)
        start, best, best_ep, since = 1, v0, 0, 0
        curve = [{"epoch": 0, "train_loss": float("nan"), "val_dice": v0, "seconds": round(time.time() - t0, 1)}]
        torch.save({"lora": lora_state(encoder), "decoder": model.mask_decoder.state_dict()},
                   os.path.join(run_dir, "best.pt"))

    # ---- train ---------------------------------------------------------------------
    model.train()
    for ep in range(start, args.epochs + 1):
        if since >= args.patience:
            break
        tset.epoch = ep
        idx = rng.permutation(len(pairs))[:per_epoch]
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(tset, idx.tolist()), batch_size=args.batch_size,
                                             shuffle=False, num_workers=workers, collate_fn=collate,
                                             pin_memory=device.type == "cuda", persistent_workers=False)
        n_b, tl, nb, te = len(loader), 0.0, 0, time.time()
        for b, batch in enumerate(loader):
            if batch is None:
                continue
            x, t, box = (z.to(device, non_blocking=True) for z in batch)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                emb = encoder(x)
                low = model(emb, box)
            loss = loss_fn(low.float(), t.float()[:, None])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl, nb = tl + loss.item(), nb + 1
            if ep == start and b in (0, 9, 49):
                el = time.time() - te
                print(f"  epoch {ep} batch {b + 1}/{n_b}: loss {loss.item():.4f} ({el:.1f} s, "
                      f"~{el / (b + 1) * n_b / 60:.0f} min per epoch)", flush=True)
        v = validate()
        improved = v > best + 1e-4
        if improved:
            best, best_ep, since = v, ep, 0
            torch.save({"lora": lora_state(encoder), "decoder": model.mask_decoder.state_dict()},
                       os.path.join(run_dir, "best.pt"))
        else:
            since += 1
        curve.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_dice": v, "seconds": round(time.time() - t0, 1)})
        print(f"epoch {ep:3d}: train loss {tl / max(nb, 1):.4f} | val Dice {v:.4f}{'  *best' if improved else ''} | "
              f"{time.time() - te:.0f} s", flush=True)
        with open(os.path.join(run_dir, "curve.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
            w.writeheader()
            w.writerows(curve)
        torch.save({"lora": lora_state(encoder), "decoder": model.mask_decoder.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep, "best": best, "best_ep": best_ep, "since": since, "curve": curve,
                    "elapsed": elapsed0 + time.time() - t0}, last)
    if since >= args.patience:
        print(f"early stop: no improvement for {args.patience} epochs")

    train_s = elapsed0 + time.time() - t0  # all submissions; includes model loading, excludes the test below
    peak = dl.peak_gpu_gb(device)

    # ---- evaluate the best weights ------------------------------------------------------
    st = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu")
    load_lora_state(encoder, st["lora"])
    model.mask_decoder.load_state_dict(st["decoder"])
    torch.save(dl.merged_state(model.mask_decoder), os.path.join(run_dir, "best_decoder.pt"))
    encoder.eval(), model.eval()
    recs = []
    test_sets = [("in-domain", args.work_dir)] + ([("zero-shot", args.zeroshot_dir)] if args.zeroshot_dir else [])
    for tname, wdir in test_sets:
        lw = live if wdir == args.work_dir else LiveWorkDir(wdir, encoder, device, autocast)
        rows = lw.rows("test")
        if tname == "in-domain":
            trained = {r["dataset"] for r in lw.manifest if r["split"] == "train"}
            rows = [r for r in rows if lw.manifest[r]["dataset"] in trained]
        n0 = len(recs)
        for r, lab, d, area in dl.predict_instances(model, lw, lw.instances(rows), device):
            m = lw.manifest[r]
            recs.append({"system": args.arm + args.tag, "name": arm_name + (f" ({args.tag.strip('_')})" if args.tag else ""),
                         "seed": args.seed, "test_set": tname, "root": m["root"], "dataset": m["dataset"],
                         "row": r, "label": lab, "area": area, "dice": d})
        print(f"test {tname}: median Dice {np.nanmedian([x['dice'] for x in recs[n0:]]):.3f} over "
              f"{len(recs) - n0} structures ({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(run_dir, "eval_instances.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    meta = {"arm": args.arm, "arm_name": arm_name, "seed": args.seed, "lr": args.lr, "rank": args.rank,
            "alpha": args.alpha, "batch_size": args.batch_size, "jitter": args.jitter, "bf16": autocast,
            "training_slices": len(pairs), "samples_per_epoch": per_epoch, "cap_per_dataset": cap,
            "mode": mode, "lora_parameters": n_lora, "trainable_parameters": n_lora,
            "train_seconds": round(train_s, 1), "peak_gpu_gb": peak,
            "epochs_run": curve[-1]["epoch"], "best_epoch": best_ep,
            "best_val_dice": best, "val_dice_before": curve[0]["val_dice"], "seconds": round(time.time() - t0, 1),
            "tag": args.tag}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    open(os.path.join(run_dir, "DONE"), "w").close()
    print(f"done: best val Dice {best:.4f} at epoch {best_ep} -> {run_dir}")


if __name__ == "__main__":
    main()
