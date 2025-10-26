from __future__ import annotations
import pickle
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
# from glob import glob
# from tqdm import tqdm
from typing import Any, Literal, Sequence
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
import os
import re
from scipy.stats import wilcoxon
from itertools import combinations
import warnings

f_path = "/Users/wangzhao/Library/CloudStorage/OneDrive-TheUniversityofQueensland/PhD/CodePractice/samri_results/"
# ds_list = ['AMOSMR',
#  'BraTS_FLAIR',
#  'BraTS_T1',
#  'BraTS_T1CE',
#  'Brain_Tumor_Dataset_Figshare',
#  'CervicalCancer',
#  'Heart',
#  'ISLES2022_ADC',
#  'ISLES2022_DWI',
#  'MSD_BrainTumour',
#  'MSD_Heart',
#  'MSD_Hippocampus',
#  'MSD_Prostate',
#  'MSK_FLASH',
#  'MSK_PD',
#  'MSK_T2',
#  'ProstateADC',
#  'ProstateT2',
#  'QIN-PROSTATE-Lesion',
#  'QIN-PROSTATE-Prostate',
#  'SpineMR',
#  'WMH_FLAIR',
#  'WMH_T1',
#  'crossmoda',
#  'totalseg_mr']

# Label projection dictionary for all datasets.
LABEL_PROJECTION= {
    'AMOSMR':{1: "Spleen", 2: "Right Kidney",  3: "Left Kidney", 4: "Gallbladder", 
              5:"Esophagus", 6:"Liver", 7: "Stomach", 8:"Aorta",
              9:"Inferior Vena Cava", 10:"Pancreas", 11:"Right Adrenal Gland",
              12:"Left Adrenal Gland", 13:"Duodenum", 14:"Right Ureter", 15:"Prostate"}, 
    'BraTS_FLAIR':{1:"Brain Tumor"}, 
    'BraTS_T1':{1:"Brain Tumor"}, 
    'BraTS_T1CE':{1:"Brain Tumor"},
    'Brain_Tumor_Dataset_Figshare': {1:"Brain Tumor"},
    'Heart': {1:"Heart"}, 
    'HipMRI':{1:"Bone", 2:"Bone", 3:"Bone", 
              4:"Bone", 5:"Bone", 6:"Bone", 7:"Bone",
              8: "Bladder", 9:"Rectum", 10:"Prostate"}, # label 1-9  background (0), bladder (3), body (1), bone (2), rectum (4) and prostate (5)
    'ISLES2022_ADC': {x:"Ischemic Stroke"  for x in range(1, 256)},
    'ISLES2022_DWI': {x:"Ischemic Stroke" for x in range(1, 256)}, 
    'MSD_BrainTumour': {1: "Edema", 2: "Non-enhancing tumor", 3: "Enhancing tumour"}, 
    'MSD_Heart':{1:"Heart"}, 
    'MSD_Hippocampus': {1: "Anterior Hippocampus", 2: "Posterior Hippocampus"},
    'MSD_Prostate': {1: "Prostate Peripheral Zone", 2:"Prostate Central Zone"}, 
    'MSK_FLASH': {1:"Bone(Femur)", 2:"Cartilage(Femur)", 3:"Bone(Tibia)", 4:"Cartilage(Tibia)", 5:"Patella", 6:"Cartilage(Patella)"},
    'MSK_PD': {1:"Bone(Femur)", 2:"Cartilage(Femur)", 3:"Bone(Tibia)", 4:"Cartilage(Tibia)", 5:"Patella", 6:"Cartilage(Patella)"},  # label 1-6 
    'MSK_T2': {1:"Bone(Femur)", 2:"Cartilage(Femur)", 3:"Bone(Tibia)", 4:"Cartilage(Tibia)", 5:"Patella", 6:"Cartilage(Patella)"},  # label 1-6 
    'OAIAKOA': {1:"Bone(Femur)", 2:"Cartilage(Femur)", 3:"Bone(Tibia)", 4:"Cartilage(Tibia)", 5:"Patella", 6:"Cartilage(Patella)"}, # label 1-6 
    'PROMISE': {1: "Prostate"},
    'Picai': {1: "Prostate"}, 
    'ProstateADC': {1: "Prostate"}, 
    'ProstateT2': {1: "Prostate"},
    'QIN-PROSTATE-Lesion': {2: "Prostate Peripheral Zone", 3: "Prostate Central Zone"},
    'QIN-PROSTATE-Prostate': {1: "Prostate"}, 
    'QUBIQ_kidney': {1: "Kidney"},
    'SpineMR': {x: "Vertebrae" for x in range(1, 256)}, 
    'WMH_FLAIR': {int(x): "White Matter Hyperintensity" for x in range(1, 256)}, 
    'WMH_T1': {int(x): "White Matter Hyperintensity" for x in range(1, 256)}, 
    'ZIB_OAI': {1:"Bone(Femur)", 2:"Cartilage(Femur)", 3:"Bone(Tibia)", 4:"Cartilage(Tibia)"},  # label 1-4 
    'crossmoda': {1: "Vestibular Schwannoma", 2: "Vestibular Schwannoma", 3: "Cochlea", 4: "Cochlea", 5: "Extra-meatal part of VS "},
    'totalseg_mr':{x: "Total body MRI" for x in range(1, 57)}, 
    'ACDC': {1: "Myocardium", 2: "Left Ventricle", 3: "Right Ventricle"},
    'CHAOS_T1': {63: "Liver", 126: "Right Kidney", 189: "Left Kidney", 252: "Spleen"}, 
    'CHAOS_T2': {63: "Liver", 126: "Right Kidney", 189: "Left Kidney", 252: "Spleen"}, 
    'CervicalCancer': {2: "Cervix Cancer"},
    'MSK_shoulder':{180: "Scapula cartilage", 225: "Humerus cartilage"}, 
    'QUBIQ_prostate': {1: "Prostate"},
}

PARTS_PROJECTION = {
    'AMOSMR':"Abdomen", 
    'BraTS_FLAIR':"Brain",
    'BraTS_T1':"Brain", 
    'BraTS_T1CE':"Brain",
    'Brain_Tumor_Dataset_Figshare': "Brain",
    'Heart': "Thorax", 
    'HipMRI':"Abdomen", 
    'ISLES2022_ADC': "Brain",
    'ISLES2022_DWI': "Brain", 
    'MSD_BrainTumour': "Brain", 
    'MSD_Heart':"Thorax", 
    'MSD_Hippocampus': "Brain",
    'MSD_Prostate': "Prostate", 
    'MSK_FLASH': "Knee",
    'MSK_PD': "Knee",  
    'MSK_T2': "Knee",  
    'OAIAKOA': "Knee", 
    'PROMISE': "Prostate",
    'Picai': "Prostate", 
    'ProstateADC': "Prostate", 
    'ProstateT2': "Prostate",
    'QIN-PROSTATE-Lesion': "Prostate",
    'QIN-PROSTATE-Prostate': "Prostate", 
    'QUBIQ_kidney': "Abdomen",
    'SpineMR': "Vertebrae", 
    'WMH_FLAIR': "Brain", 
    'WMH_T1': "Brain", 
    'ZIB_OAI': "Knee",  
    'crossmoda': "Head",
    'totalseg_mr':"Total body", 
    'ACDC': "Abdomen",
    'CHAOS_T1': "Abdomen", 
    'CHAOS_T2': "Abdomen", 
    'CervicalCancer': "Abdomen",
    'MSK_shoulder':"Shoulder", 
    'QUBIQ_prostate': "Prostate",
}
# Read pickle data
def read_data(path):
    with open(path, "rb") as f:
        x = pickle.load(f)
    return x

# Create the result dataframe for multiple models
def create_df_result_by_models(data_list, model_list):
    for model_idx, model_result in enumerate(data_list): # Result for each model
        model_name = model_list[model_idx]
        ds_names = model_result.keys()
        for ds_idx, ds_name in enumerate(ds_names):
            ds_result = model_result[ds_name]
            for data_idx, each_data in enumerate(ds_result):
                data_len = len(each_data["labels"])
                if "bp_dice" in each_data:
                    data_result = pd.DataFrame({
                        "Model":[model_name for _ in range(data_len)],
                        "Dataset": [ds_name for _ in range(data_len)], 
                        "img_path":[each_data["img_name"] for _ in range(data_len)],
                        "mask_path":[each_data["mask_name"] for _ in range(data_len)],
                        "labels":each_data["labels"],
                        "p_dice":each_data["p_dice"],
                        "b_dice":each_data["b_dice"],
                        "bp_dice":each_data["bp_dice"],
                        "p_hd":each_data["p_hd"],
                        "b_hd":each_data["b_hd"],
                        "bp_hd":each_data["bp_hd"],
                        "p_msd":each_data["p_msd"],
                        "b_msd":each_data["b_msd"],
                        "bp_msd":each_data["bp_msd"],
                        "pixel_count":each_data["pixel_count"],
                        "area_percentage":each_data["area_percentage"]
                        })
                else:
                    data_result = pd.DataFrame({
                        "Model":[model_name for _ in range(data_len)],
                        "Dataset": [ds_name for _ in range(data_len)], 
                        "img_path":[each_data["img_name"] for _ in range(data_len)],
                        "mask_path":[each_data["mask_name"] for _ in range(data_len)],
                        "labels":each_data["labels"],
                        "p_dice":each_data["p_dice"],
                        "b_dice":each_data["b_dice"],
                        "p_hd":each_data["p_hd"],
                        "b_hd":each_data["b_hd"],
                        "p_msd":each_data["p_msd"],
                        "b_msd":each_data["b_msd"],
                        "pixel_count":each_data["pixel_count"],
                        "area_percentage":each_data["area_percentage"]
                    })
                if data_idx == 0:
                    final_data = data_result
                else:
                    final_data = pd.concat([final_data, data_result], axis=0)
            if ds_idx == 0:
                final_ds = final_data
            else:
                final_ds = pd.concat([final_ds, final_data], axis=0)
        if model_idx == 0:
            final_result = final_ds
        else:
            final_result = pd.concat([final_result, final_ds], axis=0)
    return final_result

# Plot the results
def show_results(result_list, prompt="b", eval="dice", x="model", plot_style="box"):
    """Show results in boxplot or violin plot.

    Args:
        result_list (list): list of pd.DataFrame, each dataframe is the result of one experiment.
        prompt (str, optional): the prompt style, choose from "b" and "p". "b": box prompt; "p": point prompt. Defaults to "b".
        eval (str, optional): the evaluation method, choose from "dice", "hd" and "msd". "dice": Dice score; "hd": Hausdorff distance; "msd": Mean surface distance. Defaults to "dice".
        x (str, optional): The x axis label, choose from "model", "ds", and "tasks". "model": by models; "ds": by datasets; "tasks": by tasks. Defaults to "model".
        plot_style (str, optional): The plot style, choose from "box" and "violin". Defaults to "box".
    """
    # combine results
    result = pd.concat(result_list, axis=0)

    # label dictionaries
    eval_ind = {
        "dice": "Dice Score",
        "hd": "Hausdorff Distance",
        "msd": "Mean Surface Distance"
    }
    eval_lib = {
        ("dice", "p"): "p_dice", 
        ("dice", "b"): "b_dice", 
        ("hd", "p"): "p_hd", 
        ("hd", "b"): "b_hd", 
        ("msd", "p"): "p_msd", 
        ("msd", "b"): "b_msd",
        ("dice", "bp"): "bp_dice", 
        ("hd", "bp"): "bp_hd", 
        ("msd", "bp"): "bp_msd"
    }

    eval_name = eval_lib[(eval, prompt)]

    # plot
    if x == "ds":
        plt.figure(figsize=(20, 12))
        if plot_style == "box":
            sns.boxplot(
                x="Dataset", y=eval_name, data=result,
                hue="Model", palette="Set3", showfliers=False, width=0.5, whis=0.5
            )
        elif plot_style == "violin":
            sns.violinplot(
                x="Dataset", y=eval_name, data=result, inner=None,
                hue="Model", palette="Set3", width=0.5
            )
        else:
            raise ValueError("plot_style should be one of 'box' and 'violin'.")
        if eval == "dice":
            plt.legend(loc="lower right", fontsize=20)
        else:
            plt.legend(loc="upper right", fontsize=20)
        plt.xticks(rotation=90, fontsize=14)
        plt.xlabel("Dataset", fontsize=16)
        
    elif x == "model":
        plt.figure(figsize=(5, 8))
        if plot_style == "box":
            sns.boxplot(
                x="Model", y=eval_name, data=result,
                palette="Set3", showfliers=False, width=0.5, whis=0.5
            )
        elif plot_style == "violin":
            sns.violinplot(
                x="Model", y=eval_name, data=result,
                palette="Set3", width=0.5
            )
        else:
            raise ValueError("plot_style should be one of 'box' and 'violin'.")
        plt.xticks(rotation=45, fontsize=14)  # larger x ticks
        plt.xlabel("Model", fontsize=16)
        
    elif x == "tasks":
        label_order = (
            result.groupby("labels")["area_percentage"]
            .mean()
            .sort_values()
            .index
        )
        fig= plt.figure(figsize=(20, 12))
        if plot_style == "box":
            sns.boxplot(
                    x="labels", y="b_dice", data=result,
                    hue="Model", palette="Set3", showfliers=False, width=0.5, whis=0.5,
                    order=label_order
                )
        elif plot_style == "violin":
            sns.violinplot(
                    x="labels", y="b_dice", data=result,
                    hue="Model", palette="Set3", width=0.5, inner=None,
                    order=label_order
                )
        else:
            raise ValueError("plot_style should be one of 'box' and 'violin'.")
        plt.xticks(rotation=90, fontsize=14) # larger x ticks
        plt.xlabel("Task", fontsize=16)
        plt.legend(loc="lower right", fontsize=20)
    else:
        raise ValueError("x should be one of 'model', 'ds', and 'tasks'.")
    plt.yticks(fontsize=16)               # larger y ticks
      # larger X label
    plt.ylabel(eval_ind[eval], fontsize=16)                   # Y label from eval_ind
    plt.title(eval_ind[eval], fontsize=20)
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    plt.show()

# def create_csv_from_result(result, model_name, save_path):
#     result_data = read_data(result)
#     result_df = create_df_result_by_models([result_data], [model_name])
#     result_df.to_csv(save_path, index=False)


# Project labels from numbers to names.
def _normalize_to_projection_key(value: Any) -> Any:
    """Return an int when the input is a numeric string like "8.0"; otherwise the original.
    Why: Projection dict uses int keys; the dataframe may carry labels as strings cast from floats.
    """
    if pd.isna(value):
        return value

    # Fast path: already an int-like pandas NA-aware integer
    if isinstance(value, (int,)):
        return value

    # Handle strings such as "8" or "8.0" or float objects like 8.0
    try:
        f = float(value)
        if f.is_integer():
            return int(f)
        # If not an exact integer, return as-is to avoid accidental truncation
        return value
    except (TypeError, ValueError):
        return value

def project_label(data_frame: pd.DataFrame, label_projection=LABEL_PROJECTION ) -> pd.DataFrame:
    """Project numeric-like labels to names using per-dataset projection dicts.

    - Converts values like "8.0" → 8 so they match `label_projection`'s int keys.
    - Only alters rows whose `Dataset` exists in `label_projection`.
    - Leaves labels unchanged when there's no matching key.
    """
    required_cols = {"labels", "Dataset"}
    missing = required_cols.difference(data_frame.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Ensure object to hold mixed types (strings after mapping)
    data_frame["labels"] = data_frame["labels"].astype(object)

    datasets = set(data_frame["Dataset"].unique())

    for ds_name, projection in label_projection.items():
        if ds_name not in datasets:
            continue

        mask = data_frame["Dataset"].eq(ds_name)
        # Normalize to keys, then map; fallback to original when no match
        data_frame.loc[mask, "labels"] = (
            data_frame.loc[mask, "labels"]
            .apply(_normalize_to_projection_key)
            .apply(lambda k: projection.get(k, k))
        )

    return data_frame

def add_part_column(df: pd.DataFrame,
                    projection: dict = PARTS_PROJECTION,
                    default: str = "Unknown") -> pd.DataFrame:
    """
    Add a 'part' column to df by mapping df['Dataset'] using `projection`.
    Tries exact match (case-sensitive then case-insensitive), then substring match (case-insensitive).
    Unmatched -> `default`.
    """
    # Precompute case-insensitive lookup
    lower_map = {k.lower(): v for k, v in projection.items()}
    # For substring matching, try longer keys first to avoid short-key collisions
    keys_by_len_desc = sorted(projection.keys(), key=len, reverse=True)

    def project_one(name: str) -> str:
        if pd.isna(name):
            return default
        s = str(name).strip()
        # 1) exact, case-sensitive
        if s in projection:
            return projection[s]
        # 2) exact, case-insensitive
        lk = s.lower()
        if lk in lower_map:
            return lower_map[lk]
        # 3) substring match (case-insensitive)
        for k in keys_by_len_desc:
            if k.lower() in lk:
                return projection[k]
        return default

    df = df.copy()
    df["part"] = df["Dataset"].apply(project_one)
    # Optional: make it a categorical column for nicer grouping/ordering
    df["part"] = df["part"].astype("category")
    return df


# Summarize the metrics in a table for reporting.
GroupByKey = Literal["Dataset", "labels"]
EvalKey = Literal["dice", "hd", "msd"]

def _resolve_group_col(df: pd.DataFrame, group_by: GroupByKey) -> str:
    """Map user-facing `group_by` to a real column name."""
    if group_by == "labels":
        return "labels"
    if "Dataset" in df.columns:
        return "Dataset"
    raise KeyError("Expected a 'labels' or 'Dataset' column in the DataFrame.")

def _decorate_stat_cell(cell: str, mark: str) -> str:
    """
    Add `mark` at the end of the full cell string, e.g.:
      '0.51 (0.40, 0.63)' -> '0.51 (0.40, 0.63)*'
    """
    if not isinstance(cell, str) or not cell.strip():
        return cell
    return cell + mark

def summarize_metric(
    df: pd.DataFrame,
    group_by: GroupByKey = "Dataset",
    eval_key: EvalKey = "dice",
    wilcoxon=False,
) -> pd.DataFrame:
    """
    Return a formatted summary table for the requested metric.

    - Groups by [group_by, 'Model'].
    - Cells show 'median (Q1, Q3)' per group.
    - Adds 'AVG' row = mean±std over per-group medians.
    - If `wilcoxon` is True, reorders columns as
      ['SAM_vitb','MedSAM','SAMRI_box','SAMRI_bp'] and decorates cells:
        * add '*' to MedSAM / SAM_vitb when p<0.05 vs SAMRI_box
        * add '†' to MedSAM / SAM_vitb when p<0.05 vs SAMRI_bp
      matching by (group value == wil_df['label']) and
      metric column == wil_df['metric'].
    """
    if group_by not in {"Dataset", "labels"}:
        raise ValueError("group_by must be 'Dataset' or 'labels'")

    metric_map = {"dice": "b_dice", "hd": "b_hd", "msd": "b_msd"}
    if eval_key not in metric_map:
        raise ValueError("eval_key must be 'dice', 'hd', or 'msd'")
    metric_col = metric_map[eval_key]

    group_col = _resolve_group_col(df, group_by)

    # Validate required columns
    required = {group_col, "Model", metric_col}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Work on a copy; enforce numeric metric and drop non-finite values
    data = df[[group_col, "Model", metric_col]].copy()
    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric_col])

    # Compute Q1/median/Q3 per (group_col, Model)
    def _compute_stats(x: pd.Series) -> pd.Series:
        return pd.Series({
            "q1": x.quantile(0.25),
            "median": x.median(),
            "q3": x.quantile(0.75),
        })

    grouped = (
        data.groupby([group_col, "Model"], dropna=False)[metric_col]
        .apply(_compute_stats)
        .unstack()  # -> columns [q1, median, q3]
    )

    # Format as "median (Q1, Q3)"
    formatted = grouped.apply(
        lambda row: f"{row['median']:.2f} ({row['q1']:.2f}, {row['q3']:.2f})",
        axis=1,
    )

    # Pivot to table with index=group value and columns=Model
    result = formatted.unstack("Model")

    # Sort index alphabetically (case-insensitive). NaNs last.
    non_na_index = [i for i in result.index if pd.notna(i)]
    na_index = [i for i in result.index if pd.isna(i)]
    sorted_index = sorted(non_na_index, key=lambda x: str(x).casefold()) + na_index
    result = result.loc[sorted_index]

    # AVG row computed over medians
    median_only = grouped["median"].unstack("Model")
    avg_stats = median_only.agg(["mean", "std"], axis=0)
    avg_row = avg_stats.apply(lambda col: f"{col['mean']:.2f} ± {col['std']:.2f}")
    result.loc["AVG"] = avg_row

    # --- NEW: Reorder & decorate based on Wilcoxon results ---
    desired_order = ["SAM_vitb", "MedSAM", "SAMRI_box", "SAMRI_bp"]
    present = [c for c in desired_order if c in result.columns]
    others = [c for c in result.columns if c not in present]
    result = result[present + others]  # keep any extra columns at the end

    if wilcoxon:
        # Build quick lookup for significant pairs (p<0.05) per group value
        # Only rows matching this metric are considered.
        w = calculate_wilcoxon(df, by=group_by)
        # Expect wilcoxon has columns: label, metric, model_a, model_b, p_value
        needed = {"label", "metric", "model_a", "model_b", "p_value"}
        if not needed.issubset(w.columns):
            raise KeyError("`wilcoxon` must include columns: 'label','metric','model_a','model_b','p_value'")

        # Filter to this metric
        w = w[w["metric"] == metric_col].dropna(subset=["label", "model_a", "model_b", "p_value"])
        # Normalize model names for safety
        w["model_a"] = w["model_a"].astype(str)
        w["model_b"] = w["model_b"].astype(str)

        # For fast membership checks, group by label (group value)
        by_label = dict(tuple(w.groupby("label", dropna=False)))

        # Iterate over table cells for MedSAM and SAM_vitb
        for row_key in result.index:
            grp_val = row_key
            if grp_val not in by_label:
                continue

            sub = by_label[grp_val]

            # Flags: is there a significant comparison vs SAMRI_box / SAMRI_bp?
            # We mark MedSAM and SAM_vitb cells separately depending on which pair shows p<0.05
            def _has_sig_against(target_model: str, ref_model: str) -> bool:
                mask = (
                    ((sub["model_a"] == target_model) & (sub["model_b"] == ref_model)) |
                    ((sub["model_b"] == target_model) & (sub["model_a"] == ref_model))
                )
                if not mask.any():
                    return False
                return (sub.loc[mask, "p_value"] < 0.05).any()

            for target in ["MedSAM", "SAM_vitb"]:
                if target in result.columns:
                    mark = ""
                    if _has_sig_against(target, "SAMRI_box"):
                        mark += "*"
                    if _has_sig_against(target, "SAMRI_bp"):
                        mark += "†"
                    if mark:
                        # Decorate this cell (and also decorate AVG row for the same column? Only per spec for group rows)
                        cell = result.at[row_key, target]
                        result.at[row_key, target] = _decorate_stat_cell(cell, mark)

        # Note: We only mark MedSAM and SAM_vitb (per your rule).
        # SAMRI_box and SAMRI_bp columns are left unchanged.

    return result


# Plot the performance of different models on small, medium, and large objects within a specific dataset.
def small_object_show(ds, plot_style="box"):
    """Plot the performance of different models on small, medium, and large objects within a specific dataset.

    Args:
        ds (pd.DataFrame): The dataset to filter results by.
        plot_style (str, optional): The plot style, choose from "box" and "violin", refering to the box plot and the violine plot. Defaults to "box".
    """
    data = ds.copy()
    data["area_class"] = pd.qcut(data["area_percentage"], q=3, labels=["Small", "Medium", "Large"])
    plt.figure(figsize=(10, 6))
    if plot_style == "box":
        sns.boxplot(x="area_class", y="b_dice", data=data, hue="Model", palette="Set3", fliersize=0)
    elif plot_style == "violin":
        sns.violinplot(x="area_class", y="b_dice", data=data, hue="Model", palette="Set3", inner="quartiles")
    else:
        raise ValueError("Invalid plot_style. Choose 'box' or 'violin'.")
    plt.legend(loc="lower right", fontsize=16)
    plt.title(f"Area percentage", size=20)
    plt.xticks(fontsize=20) # larger x ticks
    plt.xlabel("Object Size", fontsize=16)
    plt.yticks(fontsize=16) # larger y ticks
    plt.ylabel("Dice score", fontsize=20)
    plt.grid(axis='y', which='both', linestyle='--', linewidth=0.5)
    warnings.filterwarnings('ignore')
    plt.show()
    
def area_model_table(
    ds: pd.DataFrame,
    method: Literal["mean", "median"],
    *,
    area_col: str = "area_percentage",
    model_col: str = "Model",
    value_col: str = "b_dice",
    q: int = 3,
    labels: Sequence[str] | None = ("small", "medium", "large"),
    decimals: int = 2,
) -> pd.DataFrame:
    """
    Summarize `value_col` by area bins and model, then format as strings.

    method="mean"   -> "mean ± std"      (matches Script 1)
    method="median" -> "median ± IQR"    (matches Script 2)

    Returns a pivot table with area_class as index and models as columns.
    """
    # --- checks
    required = {area_col, model_col, value_col}
    missing = required.difference(ds.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # --- bin the area into quantiles; handle tied edges & label-length mismatch gracefully
    try:
        area_class = pd.qcut(ds[area_col], q=q, labels=labels, duplicates="drop")
    except ValueError:
        # Fallback: let pandas create categories, then rename with as many labels as we have
        area_class = pd.qcut(ds[area_col], q=q, labels=None, duplicates="drop")
        if labels is not None:
            n = len(area_class.cat.categories)
            area_class = area_class.cat.rename_categories(list(labels)[:n])

    tmp = ds.copy()
    tmp["area_class"] = area_class

    # --- aggregate & format
    if method == "mean":
        grouped = (
            tmp.groupby(["area_class", model_col], observed=True)[value_col]
            .agg(["mean", "std"])
        )
        formatted = grouped.apply(
            lambda r: f"{r['mean']:.{decimals}f} ± {r['std']:.{decimals}f}", axis=1
        )

    elif method == "median":
        grouped = (
            tmp.groupby(["area_class", model_col], observed=True)[value_col]
            .agg(
                median="median",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            )
        )
        grouped["iqr"] = grouped["q3"] - grouped["q1"]
        formatted = grouped.apply(
            lambda r: f"{r['median']:.{decimals}f} ± {r['iqr']:.{decimals}f}", axis=1
        )
    else:
        raise ValueError("`method` must be 'mean' or 'median'.")

    # --- pivot to the same table shape as your scripts
    result = formatted.unstack(model_col)
    result.index.name = "area_class"
    result.columns.name = model_col
    return result

def _extract_msk_name(row):
    model = row['Model'].strip()
    path = str(row['mask_path']).strip()

    # SAMRI case: path is a tuple-like string
    if model == 'SAMRI_box' or model == 'SAMRI_bp':
        path = path.strip("(),'\" ")  # remove surrounding () or quotes if any

    # Get base filename
    base = os.path.basename(path)

    if model == 'MedSAM':
        # Remove '_1.0.npz' or similar
        msk_name = re.sub(r'(_\d+(\.\d+)+)?\.npz$', '', base)
    elif model == 'SAM_vitb' or model == 'SAMRI_box' or model == 'SAMRI_bp':
        # Remove '.nii.gz'
        msk_name = re.sub(r'\.nii\.gz$', '', base)
    else:
        msk_name = base  # fallback

    return msk_name

def calculate_wilcoxon(result_df: pd.DataFrame, by="Dataset") -> pd.DataFrame:
    
    if by not in ("Dataset", "labels"):
        raise ValueError("group_by must be either 'Dataset' or 'labels'.")

    # Apply to create 'msk_name' column
    result_df['msk_name'] = result_df.apply(_extract_msk_name, axis=1)

    # Define mask for MedSAM
    mask = result_df['Model'].str.strip() == 'MedSAM'

    # Replace characters in msk_name from position -8:-5 with 'seg'
    result_df.loc[mask, 'msk_name'] = result_df.loc[mask, 'msk_name'].apply(
        lambda x: x[: -8] + 'seg' + x[-5:] if len(x) >= 8 else x
    )


    # Define models and metrics
    models = ['MedSAM', 'SAM_vitb', 'SAMRI_box', 'SAMRI_bp']
    metrics = ['b_dice', 'b_hd', 'b_msd']

    results = []

    for model_a, model_b in combinations(models, 2):
        for metric in metrics:
            df = result_df[result_df['Model'].isin([model_a, model_b])]
            df = df.dropna(subset=[by, 'msk_name', 'Model', metric])

            pivoted = df.pivot_table(index=[by, 'msk_name'], columns='Model', values=metric)
            pivoted.columns.name = None
            pivoted.columns = pivoted.columns.astype(str)

            for label, group in pivoted.groupby(level=0):
                # Check both models exist in this group
                if model_a in group.columns and model_b in group.columns:
                    paired = group.dropna(subset=[model_a, model_b])
                    if len(paired) >= 10:
                        try:
                            stat, p = wilcoxon(paired[model_a], paired[model_b])
                            results.append({
                                'label': label,
                                'metric': metric,
                                'model_a': model_a,
                                'model_b': model_b,
                                'n': len(paired),
                                'wilcoxon_stat': stat,
                                'p_value': p
                            })
                        except ValueError:
                            results.append({
                                'label': label,
                                'metric': metric,
                                'model_a': model_a,
                                'model_b': model_b,
                                'n': len(paired),
                                'wilcoxon_stat': None,
                                'p_value': None
                            })

    # Build results DataFrame
    return pd.DataFrame(results)

# def area_performance(result, pixel_result):
#     """Plot the performance of different models on small, medium, and large objects within a specific dataset.

#     Args:
#         result (pd.DataFrame): The dataset to filter results by.
#         pixel_result (pd.DataFrame): The area data to filter results by.
#     """
#     # --- Build histogram edges from quantile bins robustly ---
#     def _get_hist_edges(df: pd.DataFrame) -> np.ndarray:
#         if "area_class" in df.columns and pd.api.types.is_categorical_dtype(df["area_class"]) \
#         and isinstance(df["area_class"].cat.categories, pd.IntervalIndex):
#             cats = df["area_class"].cat.categories
#             return np.r_[cats.left.values, cats.right.values[-1]]
#         _, edges = pd.qcut(df["area_percentage"], q=3, retbins=True, duplicates="drop")
#         return edges

#     hist_edges = _get_hist_edges(result)

#     # --- Hue order + palette for the histograms/legend ---
#     if "Dataset" not in pixel_result.columns:
#         raise KeyError("`Dataset` column is missing in pixel_result CSV.")
#     hue_order = sorted(pixel_result["Dataset"].dropna().unique().tolist())
#     pal = sns.color_palette("husl", n_colors=len(hue_order))

#     # --- Figure: left column has two rows; right column is split into zoom(top-small) + legend(bottom) ---
#     sns.set_style("whitegrid")
#     fig = plt.figure(figsize=(12, 7.5))
#     gs = GridSpec(
#         nrows=2, ncols=2,
#         width_ratios=[4.4, 2.2],
#         height_ratios=[2.2, 1.4],
#         wspace=0.15, hspace=0.10,
#     )

#     # Left column axes
#     ax_top = fig.add_subplot(gs[0, 0])
#     ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

#     # Right column is a sub-grid with two rows: [zoom (smaller), legend (larger)]
#     right_gs = gs[:, 1].subgridspec(2, 1, height_ratios=[0.5, 0.7], hspace=0.06)
#     ax_zoom = fig.add_subplot(right_gs[0, 0])
#     ax_leg  = fig.add_subplot(right_gs[1, 0])
#     ax_leg.axis("off")


#     # --- Panel A (top): lines + IQR ribbons ---
#     # Fallback: if `agg` is not provided, compute it from `result`.
#     if 'agg' not in globals():
#         needed = {"Model", "area_percentage", "b_dice"}
#         missing = needed.difference(result.columns)
#         if missing:
#             raise KeyError(f"Missing columns for building `agg`: {sorted(missing)}")
#         tmp = result[["Model", "area_percentage", "b_dice"]].dropna()
#         # Ensure we have Interval bins available
#         if "area_class" in result.columns and pd.api.types.is_categorical_dtype(result["area_class"]) \
#         and isinstance(result["area_class"].cat.categories, pd.IntervalIndex):
#             tmp = tmp.join(result["area_class"])  # reuse existing bins
#         else:
#             tmp["area_class"] = pd.qcut(tmp["area_percentage"], q=3, duplicates="drop")
#         # Summaries per (Model, bin)
#         agg = (
#             tmp.groupby(["Model", "area_class"])['b_dice']
#             .agg(
#                 median_s='median',
#                 q25_s=lambda s: s.quantile(0.25),
#                 q75_s=lambda s: s.quantile(0.75),
#             )
#             .reset_index()
#         )
#         # Numeric bin center for plotting
#         agg["bin_center"] = agg["area_class"].apply(lambda iv: (iv.left + iv.right) / 2)

#     palette = sns.color_palette("Set2", n_colors=int(agg["Model"].nunique()))
#     for (model, g), color in zip(agg.groupby("Model", sort=True), palette):
#         g = g.sort_values("bin_center")
#         ax_top.fill_between(g["bin_center"], g["q25_s"], g["q75_s"], alpha=0.15, color=color)
#         ax_top.plot(g["bin_center"], g["median_s"], label=model, color=color, linewidth=2)

#     thr = 0.80
#     ax_top.axhline(thr, ls="--", lw=1.2, color="gray", label=f"Threshold = {thr:.2f}")

#     # Shade regions where median_s >= thr by bin spans derived from `hist_edges`
#     def shade_working(ax, g, color):
#         g = g.sort_values("bin_center").reset_index(drop=True)
#         above = g["median_s"] >= thr
#         # Map each bin_center to its histogram bin index
#         idx = np.searchsorted(hist_edges, g["bin_center"].to_numpy(), side="right") - 1
#         idx = np.clip(idx, 0, len(hist_edges) - 2)
#         starts = np.where(above & ~above.shift(fill_value=False))[0]
#         ends = np.where(~above & above.shift(fill_value=False))[0] - 1
#         if len(above) and bool(above.iloc[-1]):
#             ends = np.append(ends, len(g) - 1)
#         for s, e in zip(starts, ends):
#             left = hist_edges[idx[s]]
#             right = hist_edges[idx[e] + 1]
#             ax.axvspan(left, right, color=color, alpha=0.08)

#     for (model, g), color in zip(agg.groupby("Model", sort=True), palette):
#         shade_working(ax_top, g, color)

#     ax_top.set_ylabel("Dice (median per bin)")
#     ax_top.set_title("Performance vs. Object Size")

#     # --- Panel B (bottom-left): full-range histogram (aligned bins) ---
#     sns.histplot(
#         data=pixel_result,
#         x="area_percentage",
#         hue="Dataset",
#         element="poly",
#         stat="count",          # change to 'density' for normalized distributions
#         common_norm=False,
#         ax=ax_bot,
#         legend=False,           # we'll draw a clean legend on the right panel
#     )

#     # --- Panel C (top-right): zoomed histogram (0–4%) ---

#     sns.histplot(
#         data=pixel_result,
#         x="area_percentage",
#         hue="Dataset",
#         element="poly",
#         linewidth=0.9,
#         stat="count",
#         common_norm=False,
#         legend=False,
#         ax=ax_zoom,
#         log_scale=True
#     )
#     ax_zoom.set_title("Object size (log scaled)")
#     ax_zoom.set_ylabel("Count")
#     ax_zoom.set_xlabel("")  # keep compact

#     # --- Reference line at 3.5% on all x-panels ---
#     for ax in (ax_top, ax_bot, ax_zoom):
#         ax.axvline(x=0.035, color="gray", linestyle="--", linewidth=1.2)

#     # --- X-axis formatting (keeps ORIGINAL units) ---
#     if max(result["area_percentage"].max(), pixel_result["area_percentage"].max()) <= 1.0 + 1e-9:
#         ax_bot.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
#         ax_top.set_xlabel("Object size (% of image area)")
#     else:
#         ax_top.set_xlabel("Object size (% of image area)")

#     ax_bot.set_ylabel("Count")
#     ax_bot.set_xlabel("Object size (% of image area)")

#     # ---- x-limits for shared left column (start at 0; clamp to near-max range) ---
#     xmin = 0.0
#     xmax = np.nanmax([
#         result["area_percentage"].quantile(0.995),
#         pixel_result["area_percentage"].quantile(0.995),
#     ])
#     ax_top.set_xlim(left=xmin, right=xmax)

#     # --- Left-panel legend (models) ---
#     ax_top.legend(loc="lower center", ncol=min(3, int(agg["Model"].nunique()) + 1), frameon=False)

#     # --- Panel D (bottom-right): datasets legend ---
#     handles = [Patch(facecolor=pal[i], label=label) for i, label in enumerate(hue_order)]
#     ax_leg.legend(
#         handles, [h.get_label() for h in handles],
#         title="Dataset",
#         loc="center left",
#         frameon=True,
#         fontsize=8,
#         title_fontsize=9,
#         ncol=2
#     )
#     ax_leg.set_title("")

#     plt.show()