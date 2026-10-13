#!/usr/bin/env python3
"""Extract readability and syntactic complexity features for News Bias Detection."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import textstat
    HAS_TEXTSTAT = True
    try:
        EASY_WORDS: set[str] = set(textstat.textstat._textstatistics__get_lang_easy_words())
    except Exception:
        EASY_WORDS = set()
except ImportError:
    HAS_TEXTSTAT = False
    EASY_WORDS = set()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "readability" / "outputs" / "features"

# Regex tokenizers
WORD_RE = re.compile(r"\b[A-Za-z]+(?:['’][A-Za-z]+)*\b")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
VOWELS = set("aeiouy")

FEATURE_COLUMNS = [
    # Composite Readability Metrics (Text)
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "coleman_liau_index",
    "automated_readability_index",
    "smog_index",
    "dale_chall_readability_score",
    "linsear_write_formula",
    # Underlying Component Counts & Ratios (Text)
    "char_count",
    "word_count",
    "sentence_count",
    "syllable_count",
    "avg_sentence_length",
    "avg_syllables_per_word",
    "avg_chars_per_word",
    "complex_word_count",
    "complex_word_percentage",
    "polysyllabic_per_1000_words",
    # Headline Features (Title)
    "title_word_count",
    "title_char_count",
    "title_flesch_reading_ease",
]

SYLLABLE_CACHE: dict[str, int] = {}


def count_syllables(word: str) -> int:
    """Return syllable count for a word using textstat if available or heuristic fallback."""
    cached = SYLLABLE_CACHE.get(word)
    if cached is not None:
        return cached

    if HAS_TEXTSTAT:
        try:
            s = textstat.syllable_count(word)
        except Exception:
            s = _heuristic_syllables(word)
    else:
        s = _heuristic_syllables(word)

    if len(SYLLABLE_CACHE) < 300000:
        SYLLABLE_CACHE[word] = s
    return s


def _heuristic_syllables(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    if word.endswith("e") and not (word.endswith("le") and len(word) > 2 and word[-3] not in VOWELS):
        word = word[:-1]
    count = 0
    in_vowel = False
    for char in word:
        if char in VOWELS:
            if not in_vowel:
                count += 1
                in_vowel = True
        else:
            in_vowel = False
    if word.endswith("ed") and not (word.endswith("ted") or word.endswith("ded")):
        count = max(1, count - 1)
    return max(1, count)


def compute_readability_metrics(text: Any) -> dict[str, float | int]:
    """Calculate composite readability scores and underlying metrics for an article body."""
    if pd.isna(text):
        text = ""
    text = str(text).strip()

    if not text:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
            "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0,
            "smog_index": 0.0,
            "dale_chall_readability_score": 0.0,
            "linsear_write_formula": 0.0,
            "char_count": 0,
            "word_count": 0,
            "sentence_count": 0,
            "syllable_count": 0,
            "avg_sentence_length": 0.0,
            "avg_syllables_per_word": 0.0,
            "avg_chars_per_word": 0.0,
            "complex_word_count": 0,
            "complex_word_percentage": 0.0,
            "polysyllabic_per_1000_words": 0.0,
        }

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    sentence_count = max(1, len(sentences))

    words = WORD_RE.findall(text.lower())
    word_count = len(words)
    letter_count = sum(len(w) for w in words)
    char_count = len(text)

    if word_count == 0:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
            "coleman_liau_index": 0.0,
            "automated_readability_index": 0.0,
            "smog_index": 0.0,
            "dale_chall_readability_score": 0.0,
            "linsear_write_formula": 0.0,
            "char_count": char_count,
            "word_count": 0,
            "sentence_count": sentence_count,
            "syllable_count": 0,
            "avg_sentence_length": 0.0,
            "avg_syllables_per_word": 0.0,
            "avg_chars_per_word": 0.0,
            "complex_word_count": 0,
            "complex_word_percentage": 0.0,
            "polysyllabic_per_1000_words": 0.0,
        }

    syllable_counts = [count_syllables(w) for w in words]
    total_syllables = sum(syllable_counts)
    complex_words = sum(1 for s in syllable_counts if s >= 3)
    easy_word_count = sum(1 for s in syllable_counts if s <= 2)

    # Ratios
    asl = word_count / sentence_count
    asw = total_syllables / word_count
    avg_chars = letter_count / word_count
    complex_word_pct = (complex_words / word_count) * 100.0
    polysyllabic_per_1000 = (complex_words / word_count) * 1000.0

    # 1. Flesch Reading Ease
    fre = 206.835 - (1.015 * asl) - (84.6 * asw)

    # 2. Flesch-Kincaid Grade Level
    fkgl = (0.39 * asl) + (11.8 * asw) - 15.59

    # 3. Gunning Fog Index
    fog = 0.4 * (asl + complex_word_pct)

    # 4. Coleman-Liau Index
    L = (letter_count / word_count) * 100.0
    S = (sentence_count / word_count) * 100.0
    cli = (0.0588 * L) - (0.296 * S) - 15.8

    # 5. Automated Readability Index (ARI)
    ari = (4.71 * avg_chars) + (0.5 * asl) - 21.43

    # 6. SMOG Index
    smog = 1.0430 * math.sqrt(complex_words * (30.0 / sentence_count)) + 3.1291 if sentence_count > 0 else 0.0

    # 7. Dale-Chall Readability Score
    if EASY_WORDS:
        difficult_words = sum(1 for w in words if w not in EASY_WORDS)
    else:
        difficult_words = complex_words
    diff_word_pct = (difficult_words / word_count) * 100.0
    dc = 0.1579 * diff_word_pct + 0.0496 * asl
    if diff_word_pct > 5.0:
        dc += 3.6365

    # 8. Linsear Write Formula (scaled to text)
    sample_words = min(100, word_count)
    sample_easy = easy_word_count * (sample_words / word_count)
    sample_hard = complex_words * (sample_words / word_count)
    sample_sents = max(1, sentence_count * (sample_words / word_count))
    lw_points = (sample_easy * 1 + sample_hard * 3) / sample_sents
    lw = (lw_points - 2) / 2 if lw_points > 20 else lw_points / 2

    return {
        "flesch_reading_ease": round(float(fre), 2),
        "flesch_kincaid_grade": round(float(fkgl), 2),
        "gunning_fog": round(float(fog), 2),
        "coleman_liau_index": round(float(cli), 2),
        "automated_readability_index": round(float(ari), 2),
        "smog_index": round(float(smog), 2),
        "dale_chall_readability_score": round(float(dc), 2),
        "linsear_write_formula": round(float(lw), 2),
        "char_count": int(char_count),
        "word_count": int(word_count),
        "sentence_count": int(sentence_count),
        "syllable_count": int(total_syllables),
        "avg_sentence_length": round(float(asl), 2),
        "avg_syllables_per_word": round(float(asw), 2),
        "avg_chars_per_word": round(float(avg_chars), 2),
        "complex_word_count": int(complex_words),
        "complex_word_percentage": round(float(complex_word_pct), 2),
        "polysyllabic_per_1000_words": round(float(polysyllabic_per_1000), 2),
    }


def compute_title_metrics(title: Any) -> dict[str, float | int]:
    """Calculate key length and reading ease metrics for headline / title."""
    if pd.isna(title):
        title = ""
    title = str(title).strip()

    if not title:
        return {
            "title_word_count": 0,
            "title_char_count": 0,
            "title_flesch_reading_ease": 0.0,
        }

    words = WORD_RE.findall(title.lower())
    word_count = len(words)
    char_count = len(title)

    if word_count == 0:
        return {
            "title_word_count": 0,
            "title_char_count": char_count,
            "title_flesch_reading_ease": 0.0,
        }

    syllable_counts = [count_syllables(w) for w in words]
    total_syllables = sum(syllable_counts)
    asl = float(word_count)
    asw = total_syllables / word_count
    fre = 206.835 - (1.015 * asl) - (84.6 * asw)

    return {
        "title_word_count": int(word_count),
        "title_char_count": int(char_count),
        "title_flesch_reading_ease": round(float(fre), 2),
    }


def normalize_columns(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    """Normalize common column name variants while keeping original data."""
    renames: dict[str, str] = {}
    for col in df.columns:
        key = re.sub(r"[^a-z]", "", str(col).lower())
        if key in ("text", "content"):
            renames[col] = "Text"
        elif key == "bias":
            renames[col] = "Bias"
        elif key == "title":
            renames[col] = "Title"
        elif key == "source":
            renames[col] = "Source"
    df = df.rename(columns=renames)
    if "Text" not in df.columns:
        raise ValueError(f"{file_path.name}: no 'Text' or 'content' column found.")
    return df


def find_dataset_files(input_path: Path | None) -> list[Path]:
    """Locate dataset file(s) across default project directories."""
    if input_path and input_path.exists():
        if input_path.is_file():
            return [input_path]
        return sorted(input_path.glob("*.csv"), key=lambda p: p.name.lower())

    root_combined = PROJECT_ROOT / "Combined_Dataset.csv"
    if root_combined.exists():
        return [root_combined]

    data_dir = PROJECT_ROOT / "data"
    csvs = list(data_dir.glob("*.csv"))
    if csvs:
        return sorted(csvs, key=lambda p: p.name.lower())

    ire_dir = data_dir / "IRE_Dataset"
    if ire_dir.exists():
        csvs = list(ire_dir.glob("*.csv"))
        if csvs:
            return sorted(csvs, key=lambda p: p.name.lower())

    raise FileNotFoundError("Could not find any dataset CSV in repository root or data/ folder.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=None,
                        help="Path to specific dataset CSV (or folder of CSVs).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Folder for enriched CSVs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional maximum number of rows to process (for testing).")
    args = parser.parse_args()

    files = find_dataset_files(args.input_file)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    print(f"Found {len(files)} dataset file(s) to process:")
    for f in files:
        print(f" - {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")

    for file_path in files:
        print(f"\nProcessing {file_path.name} ...")
        df = pd.read_csv(file_path, encoding="utf-8-sig", low_memory=False, nrows=args.limit)
        df = normalize_columns(df, file_path)

        if args.limit is not None and args.limit > 0:
            print(f"  [Notice] Limited to first {args.limit} rows for testing.")

        total_rows = len(df)
        article_ids = [f"{file_path.stem}_{i + 1:06d}" for i in range(total_rows)]

        texts = df["Text"].tolist()
        titles = df["Title"].tolist() if "Title" in df.columns else [""] * total_rows

        features_list: list[dict[str, float | int]] = []
        for i, (text, title) in enumerate(zip(texts, titles), start=1):
            text_metrics = compute_readability_metrics(text)
            title_metrics = compute_title_metrics(title)
            combined_row = {**text_metrics, **title_metrics}
            features_list.append(combined_row)

            if i % 5000 == 0 or i == total_rows:
                print(f"  Processed {i:,} / {total_rows:,} articles ({(i/total_rows)*100:.1f}%)")

        feature_df = pd.DataFrame(features_list, columns=FEATURE_COLUMNS)

        result = df.copy()
        if "Article_ID" in result.columns:
            result = result.drop(columns=["Article_ID"])
        result.insert(0, "Article_ID", article_ids)

        for col in FEATURE_COLUMNS:
            result[col] = feature_df[col].values

        output_path = output_dir / f"{file_path.stem}_readability.csv"
        result.to_csv(output_path, index=False)

        missing_text = int(df["Text"].isna().sum())
        summary_rows.append({
            "dataset": file_path.name,
            "input_rows": total_rows,
            "output_rows": len(result),
            "missing_text_cells": missing_text,
            "output_file": output_path.name,
        })
        print(f"  Saved enriched dataset to: {output_path} ({len(result):,} rows)")

    summary_path = output_dir / "feature_extraction_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nFeature extraction complete! Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
