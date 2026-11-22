#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Choose typical test slices for the qualitative figure of the SAMRI paper (22 Nov 2026; pool enlarged the same day).

Why: the slices of the current figure were chosen by hand, and one of them is a CT slice (QUBIQ kidney).
Here every slice comes from one rule applied to the per-structure records of the benchmark, so the figure
shows how each model usually behaves on a structure, not a favourable case.

Rule, applied to each structure type of the list CELLS:
  1. keep the test structures of that dataset and target whose size (pixels) lies between the median and the
     90th percentile of the cell, so that the structure is easy to see and is not an extreme;
  2. for each of them take the LARGEST distance between a model's Dice and that model's median Dice on the
     cell, over the four models of the figure (SAM ViT-B, MedSAM, SAMRI box, SAMRI box+point);
  3. keep the N_CAND structures with the smallest distance. Two structures of the same volume must be at
     least five slices apart, and a volume gives at most two structures when the cell has three or more
     volumes (some test sets have only two to four volumes). A cell with fewer than 40 structures (cervical
     cancer, CHAOS kidney and spleen) is used whole, without the size limits, with slices two apart.
Candidate 1 of a panel is the slice on which all four models are closest to their usual accuracy. The column
"typical" is 1 when the distance is at most TYPICAL (0.06): only those candidates should go into the paper;
the others are run for information.

Inputs (Experiment_results/): sam_vitb_bp_matched_instances.csv (SAM ViT-B, SAMRI box and box+point per
structure) and embedding_research/baselines/medsam_rerun/medsam_fixed_records.pkl (MedSAM per structure).
Output: typical_samples.csv, read by typical_samples.py on Bunya. The rec_* columns are the Dice values of
the benchmark records; the Bunya script compares its own Dice with them.

Run:  python select_typical_samples.py <path to Experiment_results> typical_samples.csv
"""
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

ER, OUT = sys.argv[1], sys.argv[2]
N_CAND = 6
TYPICAL = 0.06
MODELS = ["sam_box", "medsam", "samri_box", "samri_bp"]

# panel id, dataset, target name in the paper's tables, description shown on the preview
# (ACDC: the tables call label 2 "Left Ventricle" and label 3 "Right Ventricle", but the ACDC files use
#  0 background, 1 right ventricle cavity, 2 myocardium, 3 left ventricle cavity)
CELLS = [
    ("01_brain_tumour_T1ce",         "Brain_Tumor_Dataset_Figshare", "Brain Tumor",                 "Brain tumour, contrast-enhanced T1"),
    ("02_brain_tumour_FLAIR",        "BraTS_FLAIR",                  "Brain Tumor",                 "Brain tumour, FLAIR"),
    ("03_brain_tumour_oedema",       "MSD_BrainTumour",              "Edema",                       "Brain tumour oedema"),
    ("04_enhancing_tumour",          "MSD_BrainTumour",              "Enhancing tumour",            "Enhancing tumour"),
    ("05_white_matter_lesion",       "WMH_FLAIR",                    "White Matter Hyperintensity", "White-matter hyperintensity, FLAIR"),
    ("06_stroke_lesion_DWI",         "ISLES2022_DWI",                "Ischemic Stroke",             "Ischaemic stroke lesion, DWI"),
    ("07_stroke_lesion_ADC",         "ISLES2022_ADC",                "Ischemic Stroke",             "Ischaemic stroke lesion, ADC"),
    ("08_vestibular_schwannoma",     "crossmoda",                    "Vestibular Schwannoma",       "Vestibular schwannoma, contrast-enhanced T1"),
    ("09_hippocampus",               "MSD_Hippocampus",              "Anterior Hippocampus",        "Hippocampus (anterior part)"),
    ("10_left_atrium",               "MSD_Heart",                    "Heart",                       "Left atrium"),
    ("11_myocardium_heldout",        "ACDC",                         "Left Ventricle",              "Myocardium, ACDC label 2 (held-out dataset)"),
    ("12_left_ventricle_heldout",    "ACDC",                         "Right Ventricle",             "Left ventricle cavity, ACDC label 3 (held-out dataset)"),
    ("13_kidney_T2_heldout",         "CHAOS_T2",                     "Left Kidney",                 "Left kidney, T2 (held-out dataset)"),
    ("14_liver_T1_heldout",          "CHAOS_T1",                     "Liver",                       "Liver, T1 (held-out dataset)"),
    ("15_spleen_T2_heldout",         "CHAOS_T2",                     "Spleen",                      "Spleen, T2 (held-out dataset)"),
    ("16_liver",                     "AMOSMR",                       "Liver",                       "Liver, abdominal MRI"),
    ("17_pancreas",                  "AMOSMR",                       "Pancreas",                    "Pancreas, abdominal MRI"),
    ("18_adrenal_gland",             "AMOSMR",                       "Left Adrenal Gland",          "Left adrenal gland, abdominal MRI"),
    ("19_prostate_T2",               "ProstateT2",                   "Prostate",                    "Prostate, T2"),
    ("20_prostate_ADC",              "ProstateADC",                  "Prostate",                    "Prostate, ADC"),
    ("21_prostate_peripheral_zone",  "MSD_Prostate",                 "Prostate Peripheral Zone",    "Prostate peripheral zone"),
    ("22_prostate_heldout",          "QUBIQ_prostate",               "Prostate",                    "Prostate (held-out dataset)"),
    ("23_cervical_cancer_heldout",   "CervicalCancer",               "Cervix Cancer",               "Cervical cancer (held-out dataset)"),
    ("24_vertebra",                  "SpineMR",                      "Vertebrae",                   "Vertebra"),
    ("25_pelvic_structure",          "HipMRI",                       "Bone",                        "Pelvic structure (HipMRI)"),
    ("26_whole_body_structure",      "totalseg_mr",                  "Total body MRI",              "Whole-body MRI structure (TotalSegmentator)"),
    ("27_femoral_cartilage_DESS",    "ZIB_OAI",                      "Cartilage(Femur)",            "Femoral cartilage, DESS"),
    ("28_femoral_cartilage_FLASH",   "MSK_FLASH",                    "Cartilage(Femur)",            "Femoral cartilage, FLASH"),
    ("29_tibial_cartilage_DESS",     "ZIB_OAI",                      "Cartilage(Tibia)",            "Tibial cartilage, DESS"),
    ("30_patellar_cartilage_T2",     "MSK_T2",                       "Cartilage(Patella)",          "Patellar cartilage, T2"),
    ("31_femur_DESS",                "ZIB_OAI",                      "Bone(Femur)",                 "Femur, DESS"),
    ("32_scapula_cartilage_heldout", "MSK_shoulder",                 "Scapula cartilage",           "Scapula cartilage (held-out dataset)"),
    ("33_humeral_cartilage_heldout", "MSK_shoulder",                 "Humerus cartilage",           "Humeral cartilage (held-out dataset)"),
]

m = pd.read_csv(os.path.join(ER, "sam_vitb_bp_matched_instances.csv"), low_memory=False)
rec = pickle.load(open(os.path.join(ER, "embedding_research", "baselines", "medsam_rerun", "medsam_fixed_records.pkl"), "rb"))
rows = []
for ds, items in rec.items():
    for x in items:
        name = os.path.basename(str(x["mask_name"][0] if isinstance(x["mask_name"], (tuple, list)) else x["mask_name"]))
        stem = name[:-4].rsplit("_", 1)[0]                      # <image name without .nii.gz>_<label>.npz
        for i, lab in enumerate(x["labels"]):
            rows.append((ds, stem, float(lab), float(x["b_dice"][i])))
md = pd.DataFrame(rows, columns=["dataset", "stem", "label_value", "medsam"])
m["stem"] = m.image.str.replace(r"\.nii\.gz$", "", regex=True)
j = m.merge(md, on=["dataset", "stem", "label_value"], how="left", validate="one_to_one")
assert j.medsam.notna().all(), "MedSAM record missing for some structures"
j = j.rename(columns={"samri_box_published": "samri_box", "samri_box_point_published": "samri_bp"})
j["volume"] = j.image.str.replace(r"_img_\d+\.nii\.gz$", "", regex=True)
j["slice"] = j.image.str.extract(r"_img_(\d+)\.nii\.gz$")[0].astype(int)

out = []
for pid, ds, target, desc in CELLS:
    c = j[(j.dataset == ds) & (j.target == target)].copy()
    assert len(c), (ds, target)
    med = c[MODELS].median()
    small = len(c) < 40
    k = c.copy() if small else c[(c.pixel_count >= c.pixel_count.quantile(0.5)) & (c.pixel_count <= c.pixel_count.quantile(0.9))].copy()
    k["distance"] = (k[MODELS] - med).abs().max(axis=1)
    k = k.sort_values(["distance", "image", "label_value"])
    gap = 2 if small else 5
    per_volume = 2 if k.volume.nunique() >= 3 else N_CAND
    keep = []
    for _, r in k.iterrows():
        same = [p for p in keep if p.volume == r.volume]
        if len(same) >= per_volume or any(abs(p.slice - r.slice) < gap for p in same):
            continue
        keep.append(r)
        if len(keep) == N_CAND:
            break
    for n, r in enumerate(keep, 1):
        out.append(dict(panel=pid, candidate=n, description=desc, test_set=r.test_set, dataset=ds, image=r.image,
                        label_value=r.label_value, target_in_tables=target, pixel_count=int(r.pixel_count),
                        structures_in_cell=len(c), distance=round(float(r.distance), 4),
                        typical=int(r.distance <= TYPICAL),
                        rec_sam_box=round(float(r.sam_box), 4), rec_medsam=round(float(r.medsam), 4),
                        rec_samri_box=round(float(r.samri_box), 4), rec_samri_bp=round(float(r.samri_bp), 4),
                        med_sam_box=round(float(med.sam_box), 4), med_medsam=round(float(med.medsam), 4),
                        med_samri_box=round(float(med.samri_box), 4), med_samri_bp=round(float(med.samri_bp), 4)))
o = pd.DataFrame(out)
o.to_csv(OUT, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400); pd.set_option("display.max_colwidth", 46)
summ = o.groupby("panel").agg(candidates=("candidate", "size"), typical=("typical", "sum"), best=("distance", "min"),
                              worst=("distance", "max"), cell=("structures_in_cell", "first"),
                              volumes=("image", lambda v: v.str.replace(r"_img_\d+\.nii\.gz$", "", regex=True).nunique()),
                              sam=("med_sam_box", "first"), medsam=("med_medsam", "first"),
                              samri_box=("med_samri_box", "first"), samri_bp=("med_samri_bp", "first"))
print(summ.to_string())
print(f"\n{len(o)} structures in {o.panel.nunique()} panels -> {OUT}")
print(f"typical (distance <= {TYPICAL}): {int(o.typical.sum())} structures; every panel has at least {int(summ.typical.min())}")
