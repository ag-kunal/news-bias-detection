#!/usr/bin/env python3
"""Create local sentiment-enriched copies of every CSV in data/IRE_Dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "IRE_Dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sentiment" / "features"
POLAR_THRESHOLD = 2.0

# Word tokens are used for lexicon counts. VADER still scores full sentence text,
# including punctuation, capitalization, and many common emoticons.
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

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


def sentiment_features(text: Any, analyzer: SentimentIntensityAnalyzer) -> dict[str, float | int]:
    """Return sentence-averaged VADER scores and VADER-lexicon polar counts."""
    if pd.isna(text):
        text = ""
    text = str(text).strip()

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    sentence_scores = [analyzer.polarity_scores(s) for s in sentences]

    if sentence_scores:
        compound_mean = sum(s["compound"] for s in sentence_scores) / len(sentence_scores)
        positive_mean = sum(s["pos"] for s in sentence_scores) / len(sentence_scores)
        negative_mean = sum(s["neg"] for s in sentence_scores) / len(sentence_scores)
        neutral_mean = sum(s["neu"] for s in sentence_scores) / len(sentence_scores)
    else:
        compound_mean = positive_mean = negative_mean = neutral_mean = 0.0

    words = WORD_RE.findall(text.lower())
    lexicon_scores = [analyzer.lexicon.get(word, 0.0) for word in words]
    positive_count = sum(score >= POLAR_THRESHOLD for score in lexicon_scores)
    negative_count = sum(score <= -POLAR_THRESHOLD for score in lexicon_scores)
    total_polar_count = positive_count + negative_count

    word_count = len(words)
    scale = 1000.0 / word_count if word_count else 0.0

    return {
        "vader_compound_mean": float(compound_mean),
        "vader_positive_mean": float(positive_mean),
        "vader_negative_mean": float(negative_mean),
        "vader_neutral_mean": float(neutral_mean),
        "positive_polar_word_count": int(positive_count),
        "negative_polar_word_count": int(negative_count),
        "highly_polar_word_count": int(total_polar_count),
        "positive_polar_per_1000_words": float(positive_count * scale),
        "negative_polar_per_1000_words": float(negative_count * scale),
        "highly_polar_per_1000_words": float(total_polar_count * scale),
    }


def normalize_columns(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """Recognize common variants of the required article-body/label columns."""
    renames: dict[str, str] = {}
    for col in df.columns:
        key = re.sub(r"[^a-z]", "", str(col).lower())
        if key == "text" or key == "content":
            renames[col] = "Text"
        elif key == "bias":
            renames[col] = "Bias"
        elif key == "title":
            renames[col] = "Title"
        elif key == "source":
            renames[col] = "Source"
    df = df.rename(columns=renames)
    if "Text" not in df.columns:
        raise ValueError(f"{file_path.name}: no 'Text' (or 'content') column found. Columns: {list(df.columns)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help=f"Folder containing the five CSV datasets (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Folder for enriched CSVs (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    files = sorted(input_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if not files:
        raise FileNotFoundError(f"No .csv files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    analyzer = SentimentIntensityAnalyzer()
    summary_rows = []

    print(f"Found {len(files)} CSV file(s) in {input_dir}")
    for file_path in files:
        print(f"\nProcessing {file_path.name} ...")
        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False)
        df = normalize_columns(df, file_path)

        # Dataset-specific row IDs are stable as long as the source file's row order stays fixed.
        article_ids = [f"{file_path.stem}_{i + 1:06d}" for i in range(len(df))]
        features: list[dict[str, float | int]] = []
        texts = df["Text"].tolist()
        for i, text in enumerate(texts, start=1):
            features.append(sentiment_features(text, analyzer))
            if i % 5000 == 0:
                print(f"  processed {i:,}/{len(texts):,} articles")

        feature_df = pd.DataFrame(features, columns=FEATURE_COLUMNS)
        # Keep all original columns; create an ID only in the derived file.
        result = df.copy()
        if "Article_ID" in result.columns:
            result = result.drop(columns=["Article_ID"])
        result.insert(0, "Article_ID", article_ids)
        for col in FEATURE_COLUMNS:
            result[col] = feature_df[col].values

        output_path = output_dir / f"{file_path.stem}_sentiment.csv"
        result.to_csv(output_path, index=False)

        missing_text = int(df["Text"].isna().sum())
        summary_rows.append({
            "dataset": file_path.name,
            "input_rows": len(df),
            "output_rows": len(result),
            "missing_text_cells": missing_text,
            "output_file": output_path.name,
        })
        print(f"  saved {output_path} ({len(result):,} rows)")

    summary_path = output_dir / "feature_extraction_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nFeature extraction complete. Summary: {summary_path}")
    print("Original datasets were not modified.")


if __name__ == "__main__":
    main()
