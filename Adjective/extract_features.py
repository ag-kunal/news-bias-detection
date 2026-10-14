#!/usr/bin/env python3
"""Extract part-of-speech adjective features from news titles."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "features"
FEATURE_COLUMNS = [
    "title_adjective_count",
    "title_pos_token_count",
    "title_adjective_rate_per_100_tokens",
    "title_has_adjective",
    "title_is_valid",
]


def compute_title_features(title: Any, tokens: Iterable[Any]) -> dict[str, int | float | bool]:
    """Count ADJ-tagged alphabetic tokens and normalize by alphabetic title tokens."""
    title_text = "" if pd.isna(title) else str(title).strip()
    lexical_tokens = [token for token in tokens if token.is_alpha]
    adjective_count = sum(token.pos_ == "ADJ" for token in lexical_tokens)
    token_count = len(lexical_tokens)

    return {
        "title_adjective_count": int(adjective_count),
        "title_pos_token_count": int(token_count),
        "title_adjective_rate_per_100_tokens": (
            round(adjective_count * 100.0 / token_count, 4) if token_count else float("nan")
        ),
        "title_has_adjective": bool(adjective_count),
        "title_is_valid": bool(title_text),
    }


def normalize_columns(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """Normalize known schema variants while retaining original fields."""
    renames: dict[Any, str] = {}
    for column in df.columns:
        key = re.sub(r"[^a-z]", "", str(column).lower())
        if key == "title":
            renames[column] = "Title"
        elif key == "bias":
            renames[column] = "Bias"
        elif key == "source":
            renames[column] = "Source"
        elif key in {"storyid", "articleid"}:
            renames[column] = "StoryID" if key == "storyid" else "Article_ID"

    df = df.rename(columns=renames)
    missing = [column for column in ("Title", "Bias") if column not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name}: missing required column(s): {', '.join(missing)}")
    return df


def input_files(input_dir: Path, input_file: Path | None) -> list[Path]:
    if input_file is not None:
        path = input_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        return [path]

    directory = input_dir.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    files = sorted(directory.glob("*.csv"), key=lambda path: path.name.lower())
    if not files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-file", type=Path, default=None,
                        help="Process one CSV instead of all CSV files in --input-dir.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="en_core_web_sm",
                        help="Installed spaCy model package (default: en_core_web_sm).")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional number of initial rows per input file, for smoke tests.")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        import spacy
    except ImportError as exc:
        raise SystemExit(
            "spaCy is required. Install dependencies with: "
            "python -m pip install -r Adjective/requirements.txt"
        ) from exc

    try:
        nlp = spacy.load(args.model, disable=["parser", "ner"])
    except OSError as exc:
        raise SystemExit(
            f"Could not load spaCy model {args.model!r}. Install it with: "
            f"python -m spacy download {args.model}"
        ) from exc

    files = input_files(args.input_dir, args.input_file)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extraction_rows: list[dict[str, Any]] = []

    for file_path in files:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            low_memory=False,
            nrows=args.limit,
        )
        df = normalize_columns(df, file_path)
        titles = df["Title"].tolist()
        title_texts = ["" if pd.isna(title) else str(title) for title in titles]
        features = [
            compute_title_features(title, doc)
            for title, doc in zip(
                titles,
                nlp.pipe(title_texts, batch_size=args.batch_size),
            )
        ]

        result = df.copy()
        if "Article_ID" not in result.columns:
            result.insert(
                0,
                "Article_ID",
                [f"{file_path.stem}_{row_number:06d}" for row_number in range(1, len(result) + 1)],
            )
        feature_df = pd.DataFrame(features, columns=FEATURE_COLUMNS, index=result.index)
        for column in FEATURE_COLUMNS:
            result[column] = feature_df[column]

        output_path = output_dir / f"{file_path.stem}_adjectives.csv"
        result.to_csv(output_path, index=False)
        extraction_rows.append({
            "dataset": file_path.name,
            "input_rows": len(df),
            "output_rows": len(result),
            "nonblank_titles": int(result["title_is_valid"].sum()),
            "titles_with_lexical_tokens": int(result["title_pos_token_count"].gt(0).sum()),
            "missing_or_blank_bias": int(result["Bias"].isna().sum()),
            "model": nlp.meta.get("name", args.model),
            "model_version": nlp.meta.get("version", "unknown"),
            "spacy_version": spacy.__version__,
            "output_file": output_path.name,
        })
        print(f"Saved {output_path.name}: {len(result):,} rows")

    summary_path = output_dir / "extraction_summary.csv"
    pd.DataFrame(extraction_rows).to_csv(summary_path, index=False)
    print(f"Extraction summary: {summary_path}")


if __name__ == "__main__":
    main()