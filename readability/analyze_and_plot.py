#!/usr/bin/env python3
"""Statistical analysis and visualization of readability features by political bias."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "readability" / "outputs" / "features"
DEFAULT_PLOT_DIR = PROJECT_ROOT / "readability" / "outputs" / "plots"
CLASSES = ["left", "center", "right"]

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {"left": "#2b83ba", "center": "#abdda4", "right": "#d7191c"}


def plot_grouped_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    output_file: Path,
    ncols: int = 2,
) -> None:
    """Plot multi-panel boxplots comparing features across political bias classes."""
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, columns):
        if col not in df.columns:
            continue

        # Filter out infinite and extreme values for clean boxplot representation
        valid = df.dropna(subset=[col, "Bias"]).copy()
        q_low = valid[col].quantile(0.01)
        q_high = valid[col].quantile(0.99)
        trimmed = valid[(valid[col] >= q_low) & (valid[col] <= q_high)]

        sns.boxplot(
            data=trimmed,
            x="Bias",
            y=col,
            order=CLASSES,
            hue="Bias",
            palette=PALETTE,
            legend=False,
            showfliers=False,
            ax=ax,
            width=0.45,
            boxprops=dict(alpha=0.85),
        )

        display_name = col.replace("_", " ").title()
        ax.set_title(display_name, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Political Leaning", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=10)

    # Hide any unused subplots
    for ax in axes[len(columns):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.002)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, default=None,
                        help="Specific enriched feature CSV to analyze.")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR,
                        help=f"Folder to save plots (default: {DEFAULT_PLOT_DIR})")
    args = parser.parse_args()

    plot_dir = args.plot_dir.expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_file and args.feature_file.exists():
        input_file = args.feature_file.expanduser().resolve()
    else:
        feature_files = sorted(DEFAULT_FEATURE_DIR.glob("*_readability.csv"))
        if not feature_files:
            raise FileNotFoundError(
                f"No *_readability.csv files found in {DEFAULT_FEATURE_DIR}.\n"
                "Run 'python readability/extract_features.py' first."
            )
        input_file = feature_files[0]

    print(f"Loading enriched dataset: {input_file.name} ...")
    df = pd.read_csv(input_file, encoding="utf-8-sig", low_memory=False)

    if "Bias" not in df.columns:
        raise ValueError("Missing 'Bias' column in feature file.")

    df["Bias"] = df["Bias"].astype("string").str.strip().str.lower()
    df = df[df["Bias"].isin(CLASSES)].copy()

    print(f"Total valid labeled rows: {len(df):,}")
    print("Class distribution:")
    print(df["Bias"].value_counts())

    # Key metric groups
    composite_scores = [
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "gunning_fog",
        "automated_readability_index",
    ]

    grade_level_indices = [
        "coleman_liau_index",
        "smog_index",
        "dale_chall_readability_score",
        "linsear_write_formula",
    ]

    structural_metrics = [
        "avg_sentence_length",
        "avg_syllables_per_word",
        "avg_chars_per_word",
    ]

    complexity_metrics = [
        "complex_word_percentage",
        "polysyllabic_per_1000_words",
    ]

    headline_metrics = [
        "title_word_count",
        "title_char_count",
        "title_flesch_reading_ease",
    ]

    all_metrics = composite_scores + grade_level_indices + structural_metrics + complexity_metrics + headline_metrics

    # Statistical summary
    print("\nComputing statistical summary by political leaning ...")
    stats_df = df.groupby("Bias")[all_metrics].agg(["count", "mean", "median", "std"]).round(2)
    stats_path = plot_dir / "readability_statistical_summary.csv"
    stats_df.to_csv(stats_path)
    print(f"Statistical summary saved to: {stats_path}")

    # Generate Boxplots
    print("\nGenerating boxplot visualizations ...")
    plot_grouped_boxplots(
        df,
        composite_scores,
        "Composite Readability Scores by Political Leaning",
        "Score / Grade Level",
        plot_dir / "01_composite_readability_scores.png",
        ncols=2,
    )

    plot_grouped_boxplots(
        df,
        grade_level_indices,
        "Grade-Level Indices by Political Leaning",
        "Index / Grade Score",
        plot_dir / "02_grade_level_indices.png",
        ncols=2,
    )

    plot_grouped_boxplots(
        df,
        structural_metrics,
        "Structural & Syntactic Complexity by Political Leaning",
        "Metric Value",
        plot_dir / "03_structural_complexity.png",
        ncols=3,
    )

    plot_grouped_boxplots(
        df,
        complexity_metrics,
        "Complex Word Density by Political Leaning",
        "Proportion / Rate per 1,000 words",
        plot_dir / "04_complex_word_distribution.png",
        ncols=2,
    )

    plot_grouped_boxplots(
        df,
        headline_metrics,
        "Headline (Title) Readability & Length by Political Leaning",
        "Metric Value",
        plot_dir / "05_headline_readability.png",
        ncols=3,
    )

    print(f"\nAll plots successfully generated in: {plot_dir}/")


if __name__ == "__main__":
    main()
