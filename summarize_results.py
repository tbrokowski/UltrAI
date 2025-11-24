"""
Summarize results across folds for:
- TB deep model predictions (trainTB.py output directory)
- Pathology ML model predictions (train_pathology_ml.py output directory)

Additionally computes ULTR-AI Max fusion on test sets:
fused_proba = max(tb_proba, path_proba) per patient.

Metrics computed:
- AUROC, AUPRC, PPV, NPV, Sensitivity, Specificity
- Sensitivity/Specificity at >90% sensitivity threshold
- Sensitivity/Specificity at >70% specificity threshold
"""

import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)


def extract_numeric_id(id_str: str) -> int:
    """Extract numeric ID from various formats:
    - 'tensor(581)' -> 581
    - '25-103' -> 103
    - '581' -> 581
    """
    if isinstance(id_str, (int, float)):
        return int(id_str)
    id_str = str(id_str).strip()
    # Handle tensor format: "tensor(581)"
    match = re.search(r'tensor\((\d+)\)', id_str)
    if match:
        return int(match.group(1))
    # Handle "25-103" format
    if '-' in id_str:
        parts = id_str.split('-')
        if len(parts) == 2:
            return int(parts[1])
    # Try direct conversion
    try:
        return int(float(id_str))
    except (ValueError, TypeError):
        return None


def compute_metrics_extended(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute comprehensive metrics including AUPRC, PPV, NPV, and threshold-based metrics."""
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Basic metrics
    try:
        auroc = roc_auc_score(y_true, y_proba)
    except Exception:
        auroc = 0.0
    
    try:
        auprc = average_precision_score(y_true, y_proba)
    except Exception:
        auprc = 0.0
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    except Exception:
        specificity = 0.0
        sensitivity = 0.0
        ppv = 0.0
        npv = 0.0
    
    # Find threshold for >90% sensitivity
    thresholds = np.linspace(0, 1, 1000)
    target_sensitivity = 0.90
    best_thresh_90 = None
    best_sens_at_90 = None
    best_spec_at_90 = None
    best_sens_diff = float("inf")
    best_spec = -float("inf")
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_thresh).ravel()
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        if sens_t >= target_sensitivity:
            diff = sens_t - target_sensitivity
            if diff < best_sens_diff or (abs(diff - best_sens_diff) < 1e-9 and spec_t > best_spec):
                best_sens_diff = diff
                best_spec = spec_t
                best_thresh_90 = thresh
                best_sens_at_90 = sens_t
                best_spec_at_90 = spec_t
    sens_90_thresh = best_thresh_90
    sens_at_90sens = best_sens_at_90
    spec_at_sens90 = best_spec_at_90
    
    # Find threshold for >70% specificity
    target_specificity = 0.70
    best_thresh_70 = None
    best_spec_diff = float("inf")
    best_sens_at_70 = None
    best_spec_at_70 = None
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, y_pred_thresh).ravel()
        spec_t = tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0
        sens_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
        if spec_t >= target_specificity:
            diff = spec_t - target_specificity
            if diff < best_spec_diff or (abs(diff - best_spec_diff) < 1e-9 and sens_t > (best_sens_at_70 or -float("inf"))):
                best_spec_diff = diff
                best_thresh_70 = thresh
                best_spec_at_70 = spec_t
                best_sens_at_70 = sens_t
    spec_70_thresh = best_thresh_70
    sens_at_70spec = best_sens_at_70
    spec_at_70spec = best_spec_at_70
    
    return {
        "roc_auc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1_score": float(f1_score(y_true, y_pred)),
        "sensitivity_at_90sens": float(sens_at_90sens) if spec_at_sens90 is not None else None,
        "specificity_at_90sens": float(spec_at_sens90) if spec_at_sens90 is not None else None,
        "threshold_at_90sens": float(sens_90_thresh) if sens_90_thresh is not None else None,
        "sensitivity_at_70spec": float(sens_at_70spec) if sens_at_70spec is not None else None,
        "specificity_at_70spec": float(spec_at_70spec) if spec_at_70spec is not None else None,
        "threshold_at_70spec": float(spec_70_thresh) if spec_70_thresh is not None else None,
    }


def load_tb_preds(tb_dir: str, fold: int) -> pd.DataFrame:
    """Load TB predictions and normalize IDs to numeric format."""
    path = os.path.join(tb_dir, f"Fold_{fold}_testpredictions.csv")
    df = pd.read_csv(path)
    
    cols = {c.lower(): c for c in df.columns}
    indices_col = cols.get("indices", "indices")
    targets_col = cols.get("targets", "targets")
    proba_col = cols.get("predictions_proba", "predictions_proba")
    
    out = df[[indices_col, targets_col, proba_col]].copy()
    out.columns = ["indices", "targets", "tb_proba"]
    
    # Extract numeric IDs from tensor format or other formats
    out["indices_numeric"] = out["indices"].apply(extract_numeric_id)
    out = out[out["indices_numeric"].notna()].copy()
    out["indices_numeric"] = out["indices_numeric"].astype(int)
    
    return out[["indices_numeric", "targets", "tb_proba"]].rename(columns={"indices_numeric": "indices"})


def load_path_preds(path_dir: str, fold: int) -> pd.DataFrame:
    """Load pathology predictions and normalize IDs to numeric format."""
    path = os.path.join(path_dir, f"Pathology_Fold_{fold}_testpredictions.csv")
    df = pd.read_csv(path)
    
    cols = {c.lower(): c for c in df.columns}
    indices_col = cols.get("indices", "indices")
    targets_col = cols.get("targets", "targets")
    proba_col = cols.get("predictions_proba", "predictions_proba")
    
    out = df[[indices_col, targets_col, proba_col]].copy()
    out.columns = ["indices", "targets", "path_proba"]
    
    # Extract numeric IDs from "25-103" format or other formats
    out["indices_numeric"] = out["indices"].apply(extract_numeric_id)
    out = out[out["indices_numeric"].notna()].copy()
    out["indices_numeric"] = out["indices_numeric"].astype(int)
    
    return out[["indices_numeric", "targets", "path_proba"]].rename(columns={"indices_numeric": "indices"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tb_preds_dir",
        type=str,
        required=True,
        help="Directory where trainTB.py saved Fold_{k}_testpredictions.csv",
    )
    parser.add_argument(
        "--path_preds_dir",
        type=str,
        required=True,
        help="Directory where train_pathology_ml.py saved Pathology_Fold_{k}_testpredictions.csv",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of fold indices to include.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/trevorbrokowski/Desktop/ULTR-AI 2/benin_trust_workingfiles/predictions_path_sens_analysis",
        help="Directory to save fused results and summary metrics.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]

    summary_rows = []

    for k in folds:
        print(f"Processing fold {k}...")
        tb_df = load_tb_preds(args.tb_preds_dir, k)
        path_df = load_path_preds(args.path_preds_dir, k)
        
        print(f"  TB predictions: {len(tb_df)} samples")
        print(f"  Pathology predictions: {len(path_df)} samples")
        
        # Merge on numeric indices and targets
        merged = tb_df.merge(path_df, on=["indices", "targets"], how="inner")
        
        if merged.empty:
            print(f"[WARN] No overlap for fold {k}. Skipping.")
            print(f"  TB IDs sample: {tb_df['indices'].head(5).tolist()}")
            print(f"  Path IDs sample: {path_df['indices'].head(5).tolist()}")
            continue

        print(f"  Merged: {len(merged)} samples")
        
        # Compute ULTR-AI MAX: max(tb_proba, path_proba)
        fused_proba = np.maximum(merged["tb_proba"].values, merged["path_proba"].values)
        y_true = merged["targets"].values.astype(int)

        # Compute comprehensive metrics for each model and fused
        tb_metrics = compute_metrics_extended(y_true, merged["tb_proba"].values)
        path_metrics = compute_metrics_extended(y_true, merged["path_proba"].values)
        fused_metrics = compute_metrics_extended(y_true, fused_proba)

        # Save fused CSV
        fused_out = merged.copy()
        fused_out["fused_proba"] = fused_proba
        fused_path = os.path.join(args.output_dir, f"ULTR_AI_MAX_Fold_{k}_test.csv")
        fused_out.to_csv(fused_path, index=False)

        # Build summary row
        row = {"fold": k, "split": "test"}
        for prefix, md in [("tb", tb_metrics), ("path", path_metrics), ("fused", fused_metrics)]:
            for mname, mval in md.items():
                row[f"{prefix}_{mname}"] = mval
        summary_rows.append(row)

    if not summary_rows:
        print("No summary generated (no folds processed).")
        return

    summary_df = pd.DataFrame(summary_rows).sort_values(by="fold")
    summary_path = os.path.join(args.output_dir, "ULTR_AI_MAX_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved ULTR-AI MAX summary metrics: {summary_path}")

    # Print macro summary across folds
    def _mean_std(col: str) -> Tuple[float, float]:
        vals = summary_df[col].dropna().values
        if len(vals) == 0:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals))

    print("\n=== Summary Across Folds ===")
    for prefix in ["tb", "path", "fused"]:
        print(f"\n{prefix.upper()}:")
        auroc_mean, auroc_std = _mean_std(f"{prefix}_roc_auc")
        auprc_mean, auprc_std = _mean_std(f"{prefix}_auprc")
        sens_mean, sens_std = _mean_std(f"{prefix}_sensitivity")
        spec_mean, spec_std = _mean_std(f"{prefix}_specificity")
        ppv_mean, ppv_std = _mean_std(f"{prefix}_ppv")
        npv_mean, npv_std = _mean_std(f"{prefix}_npv")
        
        print(f"  AUROC: {auroc_mean:.3f}±{auroc_std:.3f}")
        print(f"  AUPRC: {auprc_mean:.3f}±{auprc_std:.3f}")
        print(f"  Sensitivity: {sens_mean:.3f}±{sens_std:.3f}")
        print(f"  Specificity: {spec_mean:.3f}±{spec_std:.3f}")
        print(f"  PPV: {ppv_mean:.3f}±{ppv_std:.3f}")
        print(f"  NPV: {npv_mean:.3f}±{npv_std:.3f}")
        
        # Threshold-based metrics
        sens_at_90sens_mean, sens_at_90sens_std = _mean_std(f"{prefix}_sensitivity_at_90sens")
        spec_at_90sens_mean, spec_at_90sens_std = _mean_std(f"{prefix}_specificity_at_90sens")
        if not np.isnan(sens_at_90sens_mean):
            print(f"  At >90% Sensitivity threshold:")
            print(f"    Sensitivity: {sens_at_90sens_mean:.3f}±{sens_at_90sens_std:.3f}")
            print(f"    Specificity: {spec_at_90sens_mean:.3f}±{spec_at_90sens_std:.3f}")
        
        sens_at_70spec_mean, sens_at_70spec_std = _mean_std(f"{prefix}_sensitivity_at_70spec")
        spec_at_70spec_mean, spec_at_70spec_std = _mean_std(f"{prefix}_specificity_at_70spec")
        if not np.isnan(spec_at_70spec_mean):
            print(f"  At >70% Specificity threshold:")
            print(f"    Sensitivity: {sens_at_70spec_mean:.3f}±{sens_at_70spec_std:.3f}")
            print(f"    Specificity: {spec_at_70spec_mean:.3f}±{spec_at_70spec_std:.3f}")


if __name__ == "__main__":
    main()
