#!/usr/bin/env python3
"""Train/evaluate one sentiment-only Logistic Regression baseline per dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "outputs" / "sentiment" / "features"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs" / "sentiment" / "results"
CLASSES = ["left", "center", "right"]
FEATURE_COLUMNS = [
    "vader_compound_mean",
    "vader_positive_mean",
    "vader_negative_mean",
    "vader_neutral_mean",
    "positive_polar_word_count",
    "negative_polar_word_count",
    "highly_polar_word_count",
    "positive_polar_per_1000_words",
    "negative_polar_per_1000_words",
    "highly_polar_per_1000_words",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def safe_name(filename: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURE_DIR,
                        help=f"Folder containing *_sentiment.csv files (default: {DEFAULT_FEATURE_DIR})")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
                        help=f"Folder to save evaluation results (default: {DEFAULT_RESULTS_DIR})")
    args = parser.parse_args()

    features_dir = args.features_dir.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Feature folder not found: {features_dir}\nRun sentiment/extract_features.py first."
        )

    files = sorted(features_dir.glob("*_sentiment.csv"), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No *_sentiment.csv files found in {features_dir}")

    results_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir = results_dir / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    per_class_rows: list[dict] = []

    print(f"Found {len(files)} enriched dataset(s) in {features_dir}")
    for file_path in files:
        dataset_name = file_path.name.removesuffix("_sentiment.csv")
        print(f"\nTraining on {dataset_name} ...")
        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)

        missing_cols = [c for c in ["Bias", *FEATURE_COLUMNS] if c not in df.columns]
        if missing_cols:
            print(f"  SKIPPED: missing required columns: {missing_cols}")
            summary_rows.append({"dataset": dataset_name, "status": f"skipped_missing_columns:{'|'.join(missing_cols)}"})
            continue

        labels = df["Bias"].astype("string").str.strip().str.lower()
        valid_label_mask = labels.isin(CLASSES)
        skipped_labels = int((~valid_label_mask).sum())
        work = df.loc[valid_label_mask, [*FEATURE_COLUMNS]].copy()
        y = labels.loc[valid_label_mask].astype(str)

        # These features are numeric; replace malformed/missing values defensively.
        for col in FEATURE_COLUMNS:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        counts = y.value_counts()
        absent = [label for label in CLASSES if counts.get(label, 0) == 0]
        if absent or len(counts) < 2:
            reason = f"cannot_train_missing_classes:{'|'.join(absent)}"
            print(f"  SKIPPED: {reason}")
            summary_rows.append({"dataset": dataset_name, "status": reason, "rows_total": len(df), "rows_used": len(work)})
            continue
        if int(counts.min()) < 2:
            reason = "cannot_stratify_class_with_fewer_than_2_rows"
            print(f"  SKIPPED: {reason}")
            summary_rows.append({"dataset": dataset_name, "status": reason, "rows_total": len(df), "rows_used": len(work)})
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                work, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        except ValueError as exc:
            reason = f"split_failed:{exc}"
            print(f"  SKIPPED: {reason}")
            summary_rows.append({"dataset": dataset_name, "status": reason, "rows_total": len(df), "rows_used": len(work)})
            continue

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        labels_order = CLASSES
        cm = confusion_matrix(y_test, y_pred, labels=labels_order)
        cm_df = pd.DataFrame(
            cm,
            index=[f"actual_{label}" for label in labels_order],
            columns=[f"predicted_{label}" for label in labels_order],
        )
        cm_path = confusion_dir / f"{safe_name(dataset_name)}_confusion_matrix.csv"
        cm_df.to_csv(cm_path, index=True, index_label="actual_label")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            report = classification_report(
                y_test, y_pred,
                labels=labels_order,
                target_names=labels_order,
                output_dict=True,
                zero_division=0,
            )

        for label in labels_order:
            per_class_rows.append({
                "dataset": dataset_name,
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            })

        row = {
            "dataset": dataset_name,
            "status": "ok",
            "rows_total": len(df),
            "rows_used": len(work),
            "rows_skipped_invalid_bias": skipped_labels,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "test_fraction": TEST_SIZE,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "macro_precision": precision_score(y_test, y_pred, labels=labels_order, average="macro", zero_division=0),
            "macro_recall": recall_score(y_test, y_pred, labels=labels_order, average="macro", zero_division=0),
            "macro_f1": f1_score(y_test, y_pred, labels=labels_order, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, y_pred, labels=labels_order, average="weighted", zero_division=0),
            "confusion_matrix_file": str(cm_path.relative_to(results_dir)),
        }
        summary_rows.append(row)
        print(f"  rows: {len(work):,}; macro-F1: {row['macro_f1']:.4f}; accuracy: {row['accuracy']:.4f}")

    summary_path = results_dir / "sentiment_results.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    per_class_path = results_dir / "sentiment_classification_report.csv"
    pd.DataFrame(per_class_rows).to_csv(per_class_path, index=False)

    print(f"\nSummary results: {summary_path}")
    print(f"Per-class report: {per_class_path}")
    print(f"Confusion matrices: {confusion_dir}")
    print("These results are a baseline from one stratified 80/20 holdout split; compare cautiously across datasets.")


if __name__ == "__main__":
    main()
