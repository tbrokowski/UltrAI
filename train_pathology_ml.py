"""
Train a pathology-based ML model across folds using CXR-derived features and new sensitivity labels.

Inputs:
- features_csv: CSV containing pathology feature columns and 'record_id'
- labels_csv: CSV with columns ['record_id', 'TB Label']
- splits_dir: Directory containing Fold_0.csv ... Fold_4.csv with columns ['train_ids','valid_ids','test_ids']

Outputs per fold (saved under output_dir):
- Pathology_Fold_{k}_trainpredictions.csv
- Pathology_Fold_{k}_validpredictions.csv
- Pathology_Fold_{k}_testpredictions.csv
- pathology_metrics_summary.csv (aggregate metrics across folds)
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


def read_split_file(split_csv: str) -> Tuple[List[str], List[str], List[str]]:
    df = pd.read_csv(split_csv)
    train_ids = df["train_ids"].dropna().astype(str).tolist() if "train_ids" in df.columns else []
    valid_ids = df["valid_ids"].dropna().astype(str).tolist() if "valid_ids" in df.columns else []
    test_ids = df["test_ids"].dropna().astype(str).tolist() if "test_ids" in df.columns else []
    return train_ids, valid_ids, test_ids


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    # Convert probabilities to 0/1 predictions with threshold 0.5
    y_pred = (y_proba >= 0.5).astype(int)
    try:
        auroc = roc_auc_score(y_true, y_proba)
    except Exception:
        auroc = 0.0
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    except Exception:
        specificity = 0.0
        sensitivity = 0.0

    return {
        "roc_auc": float(auroc),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred)),
    }


def save_predictions_csv(path: str, indices: np.ndarray, y_true: np.ndarray, y_proba: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "indices": indices.astype(str),
            "targets": y_true.astype(int),
            "predictions_proba": y_proba.astype(float),
            "predictions": (y_proba >= 0.5).astype(int),
        }
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def get_feature_and_target_frames(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    id_col: str = "record_id",
    target_col: str = "TB Label",
) -> pd.DataFrame:
    features = features_df.copy()
    if target_col in features.columns:
        # Avoid duplicate columns after merge which would create suffixes (_x/_y)
        features = features.drop(columns=[target_col])
    merged = features.merge(labels_df[[id_col, target_col]], on=id_col, how="inner")
    # Ensure id is string to match splits files types
    merged[id_col] = merged[id_col].astype(str)
    return merged


def run_fold(
    fold_idx: int,
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    splits_dir: str,
    output_dir: str,
    id_col: str,
    target_col: str,
    feature_exclude: List[str],
) -> dict:
    split_csv = os.path.join(splits_dir, f"Fold_{fold_idx}.csv")
    train_ids, valid_ids, test_ids = read_split_file(split_csv)

    data = get_feature_and_target_frames(features_df, labels_df, id_col=id_col, target_col=target_col)

    # Split
    train_df = data[data[id_col].isin(train_ids)].reset_index(drop=True)
    valid_df = data[data[id_col].isin(valid_ids)].reset_index(drop=True)
    test_df = data[data[id_col].isin(test_ids)].reset_index(drop=True)

    # Prepare X, y
    candidate_features = [c for c in data.columns if c not in set(feature_exclude + [target_col])]
    # Keep only numeric features
    X_train = train_df[candidate_features].select_dtypes(include=[np.number])
    X_valid = valid_df[candidate_features].select_dtypes(include=[np.number])
    X_test = test_df[candidate_features].select_dtypes(include=[np.number])
    y_train = train_df[target_col].astype(int).values
    y_valid = valid_df[target_col].astype(int).values
    y_test = test_df[target_col].astype(int).values

    # Simple, robust baseline: StandardScaler + LogisticRegression
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    # Predict probabilities
    train_proba = pipe.predict_proba(X_train)[:, 1]
    valid_proba = pipe.predict_proba(X_valid)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    # Save per-split predictions
    save_predictions_csv(
        os.path.join(output_dir, f"Pathology_Fold_{fold_idx}_trainpredictions.csv"),
        train_df[id_col].values,
        y_train,
        train_proba,
    )
    save_predictions_csv(
        os.path.join(output_dir, f"Pathology_Fold_{fold_idx}_validpredictions.csv"),
        valid_df[id_col].values,
        y_valid,
        valid_proba,
    )
    save_predictions_csv(
        os.path.join(output_dir, f"Pathology_Fold_{fold_idx}_testpredictions.csv"),
        test_df[id_col].values,
        y_test,
        test_proba,
    )

    # Metrics
    return {
        "fold": fold_idx,
        "train": compute_metrics(y_train, train_proba),
        "valid": compute_metrics(y_valid, valid_proba),
        "test": compute_metrics(y_test, test_proba),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_csv",
        type=str,
        required=True,
        help="Path to features CSV (must include 'record_id' and pathology feature columns).",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI/data/labels/sensitivity_analysis_labels.csv",
        help="Path to labels CSV with columns ['record_id','TB Label'].",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI/data/Splits",
        help="Directory containing Fold_0.csv ... Fold_4.csv",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of fold indices to run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/trevorbrokowski/Desktop/ULTR-AI 2/benin_trust_workingfiles/predictions_path_sens_analysis",
        help="Directory to save predictions per fold.",
    )
    parser.add_argument(
        "--id_col",
        type=str,
        default="record_id",
        help="ID column name present in features and splits.",
    )
    parser.add_argument(
        "--target_col", type=str, default="TB Label", help="Target column name in labels CSV."
    )
    args = parser.parse_args()

    features_df = pd.read_csv(args.features_csv)
    labels_df = pd.read_csv(args.labels_csv)

    # Drop obvious non-feature columns from training
    feature_exclude = [args.id_col]

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    all_rows = []
    for k in folds:
        metrics_k = run_fold(
            fold_idx=k,
            features_df=features_df,
            labels_df=labels_df,
            splits_dir=args.splits_dir,
            output_dir=args.output_dir,
            id_col=args.id_col,
            target_col=args.target_col,
            feature_exclude=feature_exclude,
        )
        for split in ["train", "valid", "test"]:
            row = {"fold": k, "split": split}
            row.update(metrics_k[split])
            all_rows.append(row)

    summary_df = pd.DataFrame(all_rows)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "pathology_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved metrics summary to {summary_path}")


if __name__ == "__main__":
    # Example usage:
    # python3 train_pathology_ml.py \
    #   --features_csv "/Users/trevorbrokowski/Desktop/ULTR-AI 2/benin_trust_workingfiles/CXR_Comp_Files/aiml_traincombined.csv" \
    #   --labels_csv "/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI/data/labels/sensitivity_analysis_labels.csv" \
    #   --splits_dir "/Users/trevorbrokowski/Desktop/ULTR-AI 2/UltrAI/data/Splits" \
    #   --output_dir "/Users/trevorbrokowski/Desktop/ULTR-AI 2/benin_trust_workingfiles/PredictionsPath_strat"
    main()


