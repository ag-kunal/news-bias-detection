#!/usr/bin/env python3
"""Create bias-grouped box plots for sentiment features in all enriched datasets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SENTIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SENTIMENT_DIR.parent
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "outputs" / "sentiment" / "features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sentiment" / "results" / "boxplots"
BIAS_ORDER = ["left", "center", "right"]

# Use per-1,000-word counts for polar words so longer articles do not
# automatically appear more polar simply because they contain more words.
PLOT_FEATURES = {
    "vader_compound_mean": "Mean VADER compound score",
    "positive_polar_per_1000_words": "Positive polar words per 1,000 words",
    "negative_polar_per_1000_words": "Negative polar words per 1,000 words",
    "highly_polar_per_1000_words": "Total polar words per 1,000 words",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def plot_dataset(file_path: Path, output_dir: Path) -> int:
    df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
    if "Bias" not in df.columns:
        print(f"SKIP {file_path.name}: missing Bias column")
        return 0

    df["Bias"] = df["Bias"].astype("string").str.strip().str.lower()
    df = df[df["Bias"].isin(BIAS_ORDER)].copy()
    dataset_name = file_path.name.removesuffix("_sentiment.csv")
    dataset_dir = output_dir / safe_name(dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    made = 0

    for feature, axis_label in PLOT_FEATURES.items():
        if feature not in df.columns:
            print(f"SKIP {dataset_name}: missing feature {feature}")
            continue

        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)

        values = []
        tick_labels = []
        for bias in BIAS_ORDER:
            series = df.loc[df["Bias"] == bias, feature].dropna()
            if not series.empty:
                values.append(series.to_numpy())
                tick_labels.append(f"{bias}\n(n={len(series):,})")

        if not values:
            print(f"SKIP {dataset_name}: no numeric values for {feature}")
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.boxplot(values, showfliers=True)
        ax.set_xticks(range(1, len(tick_labels) + 1), tick_labels)
        ax.set_xlabel("Existing bias label")
        ax.set_ylabel(axis_label)
        ax.set_title(f"{axis_label} by Bias\n{dataset_name}")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()

        output_file = dataset_dir / f"{safe_name(feature)}.png"
        fig.savefig(output_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_file}")
        made += 1

    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path, default=DEFAULT_FEATURE_DIR,
        help=f"Folder with *_sentiment.csv files (default: {DEFAULT_FEATURE_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Folder for plot images (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    features_dir = args.features_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not features_dir.is_dir():
        raise FileNotFoundError(
            f"Feature directory not found: {features_dir}\n"
            "Run extract_features.py first from the sentiment module."
        )

    files = sorted(features_dir.glob("*_sentiment.csv"), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No *_sentiment.csv files found in {features_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(files)} enriched dataset(s). Creating plots...")
    total = sum(plot_dataset(file_path, output_dir) for file_path in files)
    print(f"\nDone. Created {total} plot(s) in {output_dir}")
    print("Plots show feature distributions by existing Bias label; they do not prove causation or predictive usefulness.")


if __name__ == "__main__":
    main()
