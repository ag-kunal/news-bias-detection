#!/usr/bin/env python3
"""Extract punctuation features and compare them across article-bias labels.

Run from the repository root:
    python Punctuation/punctuation_analysis.py

Optional:
    python Punctuation/punctuation_analysis.py --input-dir data --output-dir outputs/punctuation
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "punctuation"

LABEL_ORDER = ["left", "center", "right"]

PUNCTUATION = {
    "period": r"\.",
    "comma": r",",
    "question_mark": r"\?",
    "exclamation_mark": r"!",
    "semicolon": r";",
    "colon": r":",
    "single_quote": r"['‘’]",
    "double_quote": r'["“”]',
    "parentheses": r"[()]",
    "square_brackets": r"[\[\]]",
    "dash_or_hyphen": r"[-‐‑‒–—]",
    "ellipsis": r"…",
}

WORD_RE = re.compile(
    r"\b[\w]+(?:['’][\w]+)*\b",
    flags=re.UNICODE,
)


def extract_punctuation_features(text: object) -> dict[str, float | int]:
    """Count punctuation and normalize counts per 1,000 word tokens."""
    if pd.isna(text):
        text = ""

    text = str(text)
    word_count = len(WORD_RE.findall(text))
    denominator = max(word_count, 1)

    features: dict[str, float | int] = {"word_count": word_count}
    total = 0

    for name, pattern in PUNCTUATION.items():
        count = len(re.findall(pattern, text))
        features[f"{name}_count"] = count
        features[f"{name}_per_1000_words"] = (
            count * 1000.0 / denominator
        )
        total += count

    features["total_punctuation_count"] = total
    features["total_punctuation_per_1000_words"] = (
        total * 1000.0 / denominator
    )

    features["exclamation_question_per_1000_words"] = (
        features["exclamation_mark_count"]
        + features["question_mark_count"]
    ) * 1000.0 / denominator

    return features


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    n = len(p_values)
    if n == 0:
        return []

    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(n, dtype=float)
    running_min = 1.0

    for reverse_idx in range(n - 1, -1, -1):
        rank = reverse_idx + 1
        idx = order[reverse_idx]
        running_min = min(
            running_min,
            values[idx] * n / rank,
        )
        adjusted_sorted[reverse_idx] = min(running_min, 1.0)

    result = np.empty(n, dtype=float)
    result[order] = adjusted_sorted
    return result.tolist()


def normalize_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Normalize common column-name variants without altering row content."""
    mapping = {}

    for col in df.columns:
        key = re.sub(r"[^a-z]", "", str(col).lower())

        if key in {"text", "content", "articletext", "body"}:
            mapping[col] = "Text"
        elif key == "bias":
            mapping[col] = "Bias"
        elif key == "title":
            mapping[col] = "Title"
        elif key == "source":
            mapping[col] = "Source"

    df = df.rename(columns=mapping)
    missing = {"Text", "Bias"} - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing required columns "
            f"{sorted(missing)}; found {list(df.columns)}"
        )

    return df


def analyze_dataset(
    csv_path: Path,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    print(f"\nProcessing {csv_path.name} ...")

    df = normalize_columns(
        pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            low_memory=False,
        ),
        csv_path,
    )

    original_rows = len(df)

    features = pd.DataFrame(
        [
            extract_punctuation_features(value)
            for value in df["Text"]
        ],
        index=df.index,
    )

    result = df.copy()

    # IDs are added to derived output only.
    result.insert(
        0,
        "Article_ID",
        [
            f"{csv_path.stem}_{i:06d}"
            for i in range(1, len(df) + 1)
        ],
    )

    result = pd.concat([result, features], axis=1)

    feature_dir = output_dir / "features"
    summary_dir = output_dir / "summaries"
    plot_dir = output_dir / "plots"

    for directory in (feature_dir, summary_dir, plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    feature_path = (
        feature_dir / f"{csv_path.stem}_punctuation.csv"
    )
    result.to_csv(feature_path, index=False)

    # Normalize labels only in the derived result used for analysis.
    result["Bias"] = (
        result["Bias"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    rate_columns = [
        f"{name}_per_1000_words"
        for name in PUNCTUATION
    ]
    rate_columns += [
        "total_punctuation_per_1000_words",
        "exclamation_question_per_1000_words",
    ]

    valid = result[
        result["Bias"].isin(LABEL_ORDER)
    ].copy()

    grouped = valid.groupby("Bias")[rate_columns].agg(
        ["count", "mean", "median", "std"]
    )
    grouped.to_csv(
        summary_dir / f"{csv_path.stem}_by_bias.csv"
    )

    # Kruskal-Wallis tests across available bias classes.
    stats_rows = []

    for feature in rate_columns:
        groups = [
            valid.loc[valid["Bias"] == label, feature]
            .dropna()
            .to_numpy()
            for label in LABEL_ORDER
        ]
        groups = [group for group in groups if len(group) > 0]
        combined_values = (
            np.concatenate(groups)
            if groups
            else np.array([])
        )

        if (
            len(groups) >= 2
            and len(combined_values) > len(groups)
        ):
            if len(np.unique(combined_values)) <= 1:
                statistic, p_value = 0.0, 1.0
            else:
                try:
                    statistic, p_value = kruskal(*groups)
                except ValueError:
                    statistic, p_value = np.nan, np.nan
        else:
            statistic, p_value = np.nan, np.nan

        stats_rows.append({
            "dataset": csv_path.name,
            "feature": feature,
            "test": "Kruskal-Wallis",
            "H_statistic": statistic,
            "p_value": p_value,
            "n_valid_rows": len(valid),
            "labels_tested": "|".join(
                label
                for label in LABEL_ORDER
                if (valid["Bias"] == label).any()
            ),
        })

    # Adjust p-values within this dataset.
    finite_indices = [
        i
        for i, row in enumerate(stats_rows)
        if np.isfinite(row["p_value"])
    ]

    adjusted = benjamini_hochberg([
        stats_rows[i]["p_value"]
        for i in finite_indices
    ])

    for i, p_adj in zip(finite_indices, adjusted):
        stats_rows[i]["p_value_BH"] = p_adj

    for row in stats_rows:
        row.setdefault("p_value_BH", np.nan)
        row["significant_BH_0_05"] = (
            bool(row["p_value_BH"] < 0.05)
            if pd.notna(row["p_value_BH"])
            else False
        )

    pd.DataFrame(stats_rows).to_csv(
        summary_dir
        / f"{csv_path.stem}_significance_tests.csv",
        index=False,
    )

    # Plot mean punctuation rates by bias label.
    means = (
        valid.groupby("Bias")[rate_columns]
        .mean()
        .reindex(LABEL_ORDER)
        .dropna(how="all")
    )

    if not means.empty:
        plot_columns = [
            f"{name}_per_1000_words"
            for name in PUNCTUATION
        ]

        ax = means[plot_columns].T.plot(
            kind="bar",
            figsize=(14, 7),
            width=0.82,
        )

        ax.set_title(
            f"Punctuation rates by bias label — {csv_path.stem}"
        )
        ax.set_xlabel("Punctuation type")
        ax.set_ylabel("Mean occurrences per 1,000 words")
        ax.set_xticklabels(
            [
                name.replace("_", " ")
                for name in PUNCTUATION
            ],
            rotation=35,
            ha="right",
        )
        ax.legend(title="Bias label")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        plt.tight_layout()
        plt.savefig(
            plot_dir / f"{csv_path.stem}_punctuation_rates.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    print(
        f"  rows preserved: {original_rows:,}; "
        f"features: {feature_path}"
    )

    manifest = {
        "dataset": csv_path.name,
        "input_rows": original_rows,
        "output_rows": len(result),
        "missing_text_cells": int(df["Text"].isna().sum()),
        "valid_bias_rows": len(valid),
        "feature_file": str(feature_path.relative_to(output_dir)),
    }

    return manifest, stats_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing dataset CSVs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for features, summaries and plots (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {input_dir}"
        )

    csv_files = sorted(
        input_dir.glob("*.csv"),
        key=lambda path: path.name.lower(),
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {input_dir}"
        )

    manifest_rows = []
    all_stats = []

    for csv_path in csv_files:
        try:
            manifest, stats = analyze_dataset(
                csv_path,
                output_dir,
            )
            manifest_rows.append(manifest)
            all_stats.extend(stats)
        except Exception as exc:
            print(f"  ERROR: {csv_path.name}: {exc}")

    if not manifest_rows:
        raise RuntimeError(
            "No datasets processed successfully; see errors above."
        )

    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "punctuation_run_manifest.csv",
        index=False,
    )
    pd.DataFrame(all_stats).to_csv(
        output_dir
        / "punctuation_significance_tests_all_datasets.csv",
        index=False,
    )

    print(
        f"\nCompleted {len(manifest_rows)} dataset(s). "
        f"Outputs: {output_dir}"
    )
    print(
        "Note: p-values indicate association, not causation. "
        "Consider effect sizes and dataset overlap before drawing conclusions."
    )


if __name__ == "__main__":
    main()
