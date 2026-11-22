#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Candidate slices for the qualitative figure with ALL comparison models (23 Nov 2026).

The first list, typical_samples.csv (196 structures, up to 6 per structure type, made by
select_typical_samples.py), was chosen with four models: SAM ViT-B, MedSAM, SAMRI box, SAMRI box+point.
It was run on Bunya (job 33298970, results in Eval_results/typical_samples/v2). This list adds the four
other comparison models of the paper - MCP-MedSAM, SAM-Med2D (fixed rerun), SAMed, Medical-SAM-Adapter -
so that one inference script can run all eight models, each in its own environment, on the same slices.

Per structure type (the 33 panels of typical_samples.csv):
  candidates 1-6   the slices of the first list, same ids; their Dice from the v2 run are in the v2_* columns
  candidates 7-12  new slices, chosen by the rule of select_typical_samples.py with the distance taken over
                   all EIGHT models' benchmark records: structures between the median and the 90th percentile
                   of the cell's size (whole cell if it has fewer than 40), smallest largest-distance first,
                   at least 5 slices (2 in a small cell) from every other candidate of the same volume, at most
                   two new candidates per volume when the cell has three or more volumes.
For every model the list gives the record Dice of the slice (rec_*) and the median Dice of the cell (med_*).

The final choice is made AFTER the all-model run, from its Dice: per structure type, the candidate whose
largest distance from the cell medians (med_* columns), over the eight models, is smallest.

Inputs
  --experiment-results  Experiment_results/: sam_vitb_bp_matched_instances.csv (SAM ViT-B, SAMRI box,
                        SAMRI box+point) and embedding_research/baselines/medsam_rerun/medsam_fixed_records.pkl
  --sam-med2d           Experiment_results/sam_med2d_fixed/sam_med2d_fixed.pkl  (fixed SAM-Med2D rerun)
  --mcp-medsam / --samed / --medsa   Bunya: /scratch/user/s4670484/Eval_results/SAMRI_comparison/<model>.pkl
  --first-list          typical_samples.csv
  --v2-dice             dice_check.csv of the v2 run
Output: a CSV that prepare_candidate_list.py (run on Bunya) completes with file paths and prompts.
Run:
  python select_all_model_candidates.py --experiment-results <ER> --sam-med2d <pkl> --mcp-medsam <pkl> \
      --samed <pkl> --medsa <pkl> --first-list typical_samples.csv --v2-dice dice_check.csv --out candidates_selected.csv
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd

N_NEW = 6
TYPICAL = 0.06
FIRST4 = ["sam_box", "medsam", "samri_box", "samri_bp"]            # models of the first list and the v2 run
OTHER4 = ["mcp_medsam", "sam_med2d", "samed", "medsa"]             # comparison models added here
MODELS = FIRST4 + OTHER4


def comparison_records(path, model):
    """test_comparison_models.py pickle -> one row per structure (dataset, image, label_value, Dice)."""
    rec = pickle.load(open(path, "rb"))
    rows = []
    for split in rec.values():
        for ds, items in split.items():
            for x in items:
                img = x["img_name"][0] if isinstance(x["img_name"], (list, tuple)) else x["img_name"]
                for lab, d in zip(x["labels"], x["dice"]):
                    rows.append((ds, os.path.basename(str(img)), float(lab), float(d)))
    df = pd.DataFrame(rows, columns=["dataset", "image", "label_value", model])
    assert not df.duplicated(["dataset", "image", "label_value"]).any(), f"{path}: duplicate structures"
    return df


def benchmark_table(args):
    """All test structures with the record Dice of the eight models (SAM / SAMRI / MedSAM as in
    select_typical_samples.py)."""
    er = args.experiment_results
    m = pd.read_csv(os.path.join(er, "sam_vitb_bp_matched_instances.csv"), low_memory=False)
    rec = pickle.load(open(os.path.join(er, "embedding_research", "baselines", "medsam_rerun",
                                        "medsam_fixed_records.pkl"), "rb"))
    rows = []
    for ds, items in rec.items():
        for x in items:
            name = os.path.basename(str(x["mask_name"][0] if isinstance(x["mask_name"], (tuple, list)) else x["mask_name"]))
            stem = name[:-4].rsplit("_", 1)[0]
            for i, lab in enumerate(x["labels"]):
                rows.append((ds, stem, float(lab), float(x["b_dice"][i])))
    md = pd.DataFrame(rows, columns=["dataset", "stem", "label_value", "medsam"])
    m["stem"] = m.image.str.replace(r"\.nii\.gz$", "", regex=True)
    j = m.merge(md, on=["dataset", "stem", "label_value"], how="left", validate="one_to_one")
    j = j.rename(columns={"samri_box_published": "samri_box", "samri_box_point_published": "samri_bp"})
    for model, path in (("mcp_medsam", args.mcp_medsam), ("sam_med2d", args.sam_med2d),
                        ("samed", args.samed), ("medsa", args.medsa)):
        j = j.merge(comparison_records(path, model), on=["dataset", "image", "label_value"], how="left",
                    validate="one_to_one")
    missing = {k: int(j[k].isna().sum()) for k in MODELS if j[k].isna().any()}
    assert not missing, f"record Dice missing for some structures: {missing}"
    j["volume"] = j.image.str.replace(r"_img_\d+\.nii\.gz$", "", regex=True)
    j["slice"] = j.image.str.extract(r"_img_(\d+)\.nii\.gz$")[0].astype(int)
    return j


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--experiment-results", "--sam-med2d", "--mcp-medsam", "--samed", "--medsa", "--first-list",
              "--v2-dice", "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()

    j = benchmark_table(args)
    first = pd.read_csv(args.first_list)
    v2 = pd.read_csv(args.v2_dice).pivot(index="structure", columns="model", values="dice")
    panels = first.drop_duplicates("panel")[["panel", "description", "dataset", "target_in_tables"]]

    out = []
    for _, p in panels.iterrows():
        cell = j[(j.dataset == p.dataset) & (j.target == p.target_in_tables)].copy()
        med = cell[MODELS].median()
        cell["rec_distance8"] = (cell[MODELS] - med).abs().max(axis=1)
        small = len(cell) < 40
        gap = 2 if small else 5
        per_volume = 2 if cell.volume.nunique() >= 3 else N_NEW

        old = first[first.panel == p.panel].merge(cell, on=["image", "label_value"], how="left",
                                                  suffixes=("", "_bench"), validate="one_to_one")
        assert old[MODELS].notna().all().all(), f"{p.panel}: a first-list structure is not in the benchmark table"
        keep = [(r.volume, r.slice) for r in old.itertuples()]
        old_keys = set(zip(old.image, old.label_value))

        pool = cell if small else cell[(cell.pixel_count >= cell.pixel_count.quantile(0.5))
                                       & (cell.pixel_count <= cell.pixel_count.quantile(0.9))]
        pool = pool.sort_values(["rec_distance8", "image", "label_value"])
        new = []
        for r in pool.itertuples():
            if (r.image, r.label_value) in old_keys:
                continue
            if sum(1 for n in new if n.volume == r.volume) >= per_volume:
                continue
            if any(v == r.volume and abs(s - r.slice) < gap for v, s in keep):
                continue
            new.append(r)
            keep.append((r.volume, r.slice))
            if len(new) == N_NEW:
                break

        rows = [("tested_v2", int(r.candidate), r) for r in old.itertuples()]
        rows += [("new", 6 + n, r) for n, r in enumerate(new, 1)]
        for source, cand, r in rows:
            sid = f"{p.panel}_c{cand}"
            row = dict(sample_id=sid, panel=p.panel, candidate=cand, source=source, description=p.description,
                       test_set=r.test_set, dataset=p.dataset, image=r.image, label_value=r.label_value,
                       target_in_tables=p.target_in_tables, pixel_count=int(r.pixel_count),
                       structures_in_cell=len(cell), rec_distance8=round(float(r.rec_distance8), 4),
                       typical8_in_records=int(r.rec_distance8 <= TYPICAL))
            row.update({f"rec_{k}": round(float(getattr(r, k)), 4) for k in MODELS})
            row.update({f"med_{k}": round(float(med[k]), 4) for k in MODELS})
            row.update({f"v2_{k}": (round(float(v2.loc[sid, k]), 4) if sid in v2.index else np.nan) for k in FIRST4})
            out.append(row)

    o = pd.DataFrame(out)
    o.to_csv(args.out, index=False)
    summ = o.groupby("panel").agg(tested=("source", lambda s: int((s == "tested_v2").sum())),
                                  new=("source", lambda s: int((s == "new").sum())),
                                  cell=("structures_in_cell", "first"),
                                  best_in_records=("rec_distance8", "min"))
    pd.set_option("display.width", 200)
    print(summ.to_string())
    print(f"\n{len(o)} candidates in {o.panel.nunique()} structure types ({int((o.source == 'new').sum())} new) -> {args.out}")


if __name__ == "__main__":
    main()
