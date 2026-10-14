#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 of the embedding experiments (GPU): run ONE frozen image encoder under
ONE preprocessing pipeline on the manifest slices and store its features.

Outputs in <work-dir>/features/<encoder>__<pipeline>[__noise<s>]/:
  neck.npy          (N, 256, 64, 64) fp16  - the embedding SAM's decoder sees,
                    for the rows in neck_rows.npy (probes, and the CKA 'neck')
  layer_tokens.npy  (M, L, T, D) fp16      - T sampled tokens from every layer
                    ('embed' + each block) for the test rows in token_rows.npy
                    (CKA). Token positions (token_pos.npy, token_inbox.npy)
                    depend only on the pipeline, so every encoder run under the
                    same pipeline is sampled at the same image locations.
  layer_names.json, meta.json, DONE (written last; a finished condition is
  skipped on resubmission unless --overwrite)

Checks run BEFORE the bulk extraction (read them in the first minutes of the
job log):
  1. GPU visible (unless --device cpu), interpreter and torch build printed.
  2. Decoder sanity: the model's own prompt encoder + mask decoder on the first
     --sanity-n test instances with a tight box prompt, Dice on native pixels.
     Expect roughly 0.7-0.9 for SAM / MedSAM under their own pipeline; a
     value near 0 means preprocessing or checkpoint loading is wrong.
  3. Stored-embedding check (sam_vit_b + sam pipeline, with
     --stored-embedding-root): compares the neck with SAMRI's precomputed
     training embeddings for the same slices; cosine should be ~1.0.

Usage (from the repo root; see run_embedding_experiments.sh for the Slurm job):
  python embedding_assess/extract_features.py \
      --work-dir /scratch/user/s4670484/Eval_results/embedding_research/ \
      --ckpt-dir /scratch/user/s4670484/pretrained_ckpt/ \
      --encoder sam_vit_b --pipeline sam \
      --stored-embedding-root /scratch/user/s4670484/Datasets/Embedding_train/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emb_common as ec  # noqa: E402


class SliceSet(torch.utils.data.Dataset):
    def __init__(self, manifest, rows, pipeline, noise_std, seed):
        self.m, self.rows, self.pl, self.noise, self.seed = manifest, rows, pipeline, noise_std, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        raw = ec.load_slice(self.m[r]["img_path"])
        rng = np.random.default_rng([self.seed, r]) if self.noise else None
        x, _ = ec.preprocess(raw, self.pl, self.noise, rng)
        return x, k


def decoder_sanity(sam, encoder, manifest, pipeline, n, device):
    test_rows = [r["row"] for r in manifest if r["split"] == "test"]
    scores, done = [], 0
    for r in test_rows:
        if done >= n:
            break
        raw, seg = ec.load_slice(manifest[r]["img_path"], manifest[r]["seg_path"])
        x, geom = ec.preprocess(raw, pipeline)
        with torch.no_grad():
            emb = encoder(x[None].to(device))
            for lab in ec.slice_labels(seg):
                if done >= n:
                    break
                m = seg == lab
                box = torch.as_tensor(ec.box_to_input(ec.box_from_mask(m), geom)[None],
                                      dtype=torch.float32, device=device)
                sparse, dense = sam.prompt_encoder(points=None, boxes=box, masks=None)
                low, _ = sam.mask_decoder(image_embeddings=emb, image_pe=sam.prompt_encoder.get_dense_pe(),
                                          sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                          multimask_output=False)
                pred = ec.grid_to_native(low[0, 0].float().cpu(), geom) > 0
                scores.append(ec.dice(pred, m))
                done += 1
    return scores


def stored_embedding_check(encoder, manifest, root, device, k=5):
    """Recompute the neck for the first k training rows and compare with
    SAMRI's stored embeddings (process_embedding.py) for the same slices."""
    out = []
    train_rows = [r["row"] for r in manifest if r["split"] == "train"][:k]
    for r in train_rows:
        row = manifest[r]
        x, _ = ec.preprocess(ec.load_slice(row["img_path"]), "sam")
        with torch.no_grad():
            neck = encoder(x[None].to(device))[0].cpu().numpy()
        name = os.path.basename(row["img_path"])[:-len(".nii.gz")]
        path = os.path.join(root, row["dataset"], name + ".npz")
        if not os.path.isfile(path):
            out.append(f"    {name}: stored embedding not found at {path}")
            continue
        ref = np.asarray(np.load(path)["img"], np.float32).reshape(-1)
        cur = neck.astype(np.float32).reshape(-1)
        if ref.size != cur.size:
            out.append(f"    {name}: stored shape differs ({ref.size} vs {cur.size} values)")
            continue
        cos = float(ref @ cur / (np.linalg.norm(ref) * np.linalg.norm(cur) + 1e-12))
        out.append(f"    {name}: cosine {cos:.5f}, max |diff| {np.abs(ref - cur).max():.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--encoder", required=True, choices=list(ec.ENCODERS))
    ap.add_argument("--pipeline", required=True, choices=ec.PIPELINES)
    ap.add_argument("--noise-std", type=float, default=0.0,
                    help="Gaussian input noise in grey levels (0-255 scale); for the CKA noise ceiling")
    ap.add_argument("--rows", default="all", choices=["all", "test"],
                    help="manifest rows to store the neck for (test = CKA-only conditions)")
    ap.add_argument("--layers", default="all", choices=["all", "neck"],
                    help="'all' also stores sampled tokens of every layer for the test rows (CKA)")
    ap.add_argument("--tokens", type=int, default=128, help="tokens sampled per slice for CKA")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows (quick test)")
    ap.add_argument("--sanity-n", type=int, default=16)
    ap.add_argument("--stored-embedding-root", default=None,
                    help="SAMRI's Embedding_train/ folder, for the consistency check")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    # ---- 1. preflight -------------------------------------------------------
    print(sys.executable, "| torch", torch.__version__, "| hip", torch.version.hip, "| cuda", torch.version.cuda)
    if args.device != "cpu":
        assert torch.cuda.is_available(), "no GPU visible - wrong conda env for this card?"
        print("device:", torch.cuda.get_device_name(0), "| visible GPUs:", torch.cuda.device_count())
    device = torch.device(args.device)
    workers = args.workers if args.workers is not None else max(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1, 0)

    cond = ec.cond_name(args.encoder, args.pipeline, args.noise_std)
    out = os.path.join(args.work_dir, "features", cond)
    if os.path.isfile(os.path.join(out, "DONE")) and not args.overwrite:
        print(f"{cond}: already finished ({out}/DONE) - skipping. Use --overwrite to redo.")
        return
    os.makedirs(out, exist_ok=True)

    manifest = ec.read_manifest(os.path.join(args.work_dir, "manifest.csv"))
    geo = np.load(os.path.join(args.work_dir, f"geometry_{args.pipeline}.npz"))
    neck_rows = [r["row"] for r in manifest if args.rows == "all" or r["split"] == "test"]
    if args.limit:
        neck_rows = neck_rows[:args.limit]
    tok_rows = [r for r in neck_rows if manifest[r]["split"] == "test"]
    print(f"{cond}: neck for {len(neck_rows)} rows, CKA tokens for {len(tok_rows)} test rows "
          f"({'all layers' if args.layers == 'all' else 'neck only'}), {workers} loader workers")

    sam, encoder = ec.load_model(args.encoder, args.ckpt_dir, device, args.seed)

    # ---- 2. decoder sanity / 3. stored-embedding check ------------------------
    if sam is not None and args.sanity_n:
        s = decoder_sanity(sam, encoder, manifest, args.pipeline, args.sanity_n, device)
        print(f"[sanity] {args.encoder} own decoder, box prompt, {args.pipeline} pipeline: "
              f"mean Dice {np.nanmean(s):.3f} over {len(s)} test instances (min {np.nanmin(s):.3f})", flush=True)
    if args.stored_embedding_root and (args.encoder, args.pipeline, args.noise_std) == ("sam_vit_b", "sam", 0.0):
        print("[check] SAM ViT-B neck vs SAMRI's stored training embeddings:")
        print("\n".join(stored_embedding_check(encoder, manifest, args.stored_embedding_root, device)), flush=True)

    # ---- token positions (pipeline-dependent only) ---------------------------
    if tok_rows:
        boxes_by_row = {}
        for r, bg in zip(geo["inst_row"], geo["inst_boxgrid"]):
            boxes_by_row.setdefault(int(r), []).append(bg)
        pos, inbox = zip(*[ec.token_positions(boxes_by_row.get(r, []), geo["row_valid"][r], args.tokens,
                                              np.random.default_rng([args.seed, r, 1])) for r in tok_rows])
        np.save(os.path.join(out, "token_rows.npy"), np.array(tok_rows, np.int32))
        np.save(os.path.join(out, "token_pos.npy"), np.stack(pos))
        np.save(os.path.join(out, "token_inbox.npy"), np.stack(inbox))
    tap, tok_index = None, {}
    if tok_rows and args.layers == "all":  # neck tokens come from neck.npy; other layers stored here
        pos_t = torch.as_tensor(np.stack(pos), dtype=torch.long)
        tok_index = {r: i for i, r in enumerate(tok_rows)}
        tap = ec.LayerTap(encoder)
        D = encoder.blocks[0].attn.qkv.in_features
        layer_mm = np.lib.format.open_memmap(os.path.join(out, "layer_tokens.npy"), mode="w+", dtype=np.float16,
                                             shape=(len(tok_rows), len(tap.names), args.tokens, D))
        with open(os.path.join(out, "layer_names.json"), "w") as f:
            json.dump(tap.names, f)

    np.save(os.path.join(out, "neck_rows.npy"), np.array(neck_rows, np.int32))
    neck_mm = np.lib.format.open_memmap(os.path.join(out, "neck.npy"), mode="w+", dtype=np.float16,
                                        shape=(len(neck_rows), 256, ec.GRID, ec.GRID))

    loader = torch.utils.data.DataLoader(SliceSet(manifest, neck_rows, args.pipeline, args.noise_std, args.seed),
                                         batch_size=args.batch_size, num_workers=workers, shuffle=False)
    n_batches, t1 = len(loader), time.time()
    for b, (x, ks) in enumerate(loader):
        with torch.no_grad():
            neck = encoder(x.to(device))
        neck_np = neck.half().cpu().numpy()
        for j, k in enumerate(ks.tolist()):
            r = neck_rows[k]
            neck_mm[k] = neck_np[j]
            if r in tok_index:
                i = tok_index[r]
                p = pos_t[i].to(device)
                layer_mm[i] = torch.stack([tap.out[nm][j].reshape(-1, D)[p] for nm in tap.names]).half().cpu().numpy()
        if b == 0 or (b + 1) % 50 == 0 or b + 1 == n_batches:
            el = time.time() - t1
            print(f"  batch {b + 1}/{n_batches}  {el:.0f} s elapsed, ~{el / (b + 1) * (n_batches - b - 1):.0f} s left",
                  flush=True)
    neck_mm.flush()
    if tap is not None:
        layer_mm.flush()
        tap.remove()

    meta = {"encoder": args.encoder, "pipeline": args.pipeline, "noise_std": args.noise_std,
            "n_neck_rows": len(neck_rows), "n_token_rows": len(tok_rows), "tokens": args.tokens,
            "ckpt": ec.ENCODERS[args.encoder][1], "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if args.device != "cpu" else "cpu",
            "seconds": round(time.time() - t0, 1), "limit": args.limit}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if not args.limit:
        open(os.path.join(out, "DONE"), "w").close()
    print(f"{cond}: finished in {time.time() - t0:.0f} s -> {out}")



if __name__ == "__main__":
    main()
