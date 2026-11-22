#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete the all-model candidate list on Bunya: file paths, image size and the prompts every model uses.

Reads the output of select_all_model_candidates.py, finds every image and the mask the benchmark paired with
it (typical_samples.resolve_files: NiiDataset pairs the sorted *_img_* and *_seg_* files by position), checks
that the structure is there with the pixel count of the records, and adds:

  img_path, seg_path        absolute paths
  height, width             slice size
  box_x0 box_y0 box_x1 box_y1   tight box of the structure, utils.utils.gen_bboxes(gt, jitter=0), pixel
                            coordinates (x = column, y = row), max corner inclusive - the box of test_vis.py
  point_x point_y           one point, utils.utils.gen_points(gt) right after random.seed(f"{seed}/{sample_id}");
                            for candidates 1-6 it is the point the v2 run gave SAMRI box+point

Every model reads the same slice the same way as in the benchmark (typical_samples.load_slice: first plane
of the NIfTI, structure = mask == label_value; SAM-style input = slice min-max to 0-255, three channels).
A model that takes its prompt in another frame (MedSAM 1024, SAM-Med2D 256, MCP-MedSAM 256) scales this
box as its own benchmark script does. A point-prompted model (SAMRI box+point, Medical-SAM-Adapter) gets the
same point: call random.seed(f"{seed}/{sample_id}") just before its gen_points call, or pass point_x/point_y.

Run on Bunya (login node is enough, about a minute):
  python prepare_candidate_list.py --candidates candidates_selected.csv \
      --data-root /scratch/user/s4670484/Datasets --repo-root /scratch/user/s4670484/SAMRI \
      --out /scratch/user/s4670484/SAMRI/evaluation/typical_samples/typical_samples_all_models.csv
"""
import argparse
import csv
import os
import random
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="seed of the point; 0 = the v2 run")
    args = ap.parse_args()

    sys.path.insert(0, args.repo_root)
    sys.path.insert(0, os.path.join(args.repo_root, "evaluation", "typical_samples"))
    import numpy as np
    from typical_samples import read_samples, resolve_files, load_slice
    from utils.utils import gen_bboxes, gen_points

    rows = read_samples(args.candidates)          # checks the columns, sets r["key"] = <panel>_c<candidate>
    resolve_files(rows, args.data_root)           # stops if an image or its mask is missing
    raw = {r["sample_id"]: r for r in csv.DictReader(open(args.candidates, newline=""))}
    out, problems = [], []
    for r in rows:
        if r["key"] != raw[r["key"]]["sample_id"]:
            problems.append(f"{r['key']}: sample_id does not match panel and candidate")
            continue
        img, gt = load_slice(r)
        n = int(gt.sum())
        if r["pixel_count"] and n != r["pixel_count"]:
            problems.append(f"{r['key']}: {n} pixels, the records say {r['pixel_count']}")
        box = [int(v) for v in gen_bboxes(gt, jitter=0)]
        random.seed(f"{args.seed}/{r['key']}")    # as typical_samples.predict_sam
        px, py = [int(v) for v in gen_points(gt)[0]]
        assert gt[py, px], f"{r['key']}: the point is outside the structure"
        row = dict(raw[r["key"]])
        row.update(img_path=r["img_path"], seg_path=r["seg_path"], height=gt.shape[0], width=gt.shape[1],
                   box_x0=box[0], box_y0=box[1], box_x1=box[2], box_y1=box[3], point_x=px, point_y=py,
                   point_seed=f"{args.seed}/{r['key']}")
        out.append(row)
    if problems:
        print("\n".join("  " + p for p in problems))
        sys.exit(f"{len(problems)} problem(s) - list not written")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} candidates in {len({r['panel'] for r in out})} structure types; all images, masks and "
          f"structures found; pixel counts match the records -> {args.out}")


if __name__ == "__main__":
    main()
