#!/usr/bin/env python3
"""OPTIONAL comparison backend: article-level fallacy signals from a zero-shot NLI model.

The papers (MAFALDA, FAINA) report that zero-shot models score very poorly on this task, so this is kept only
as a baseline to compare against the supervised extractor (extract_features.py). It is slow on CPU
(12 labels x sentences x articles NLI passes); use --limit and --max-sentences.

This is a feasibility baseline for Stage 2, not a validated fallacy detector.
It scores sentences against a documented fallacy taxonomy with a zero-shot
classifier. Scores are model hypotheses, not human annotations or proof of bias.

Example:
    python fallacy_detection/extract_features_zeroshot.py --input-dir data --limit 100

Full run (downloads the model on first use):
    python fallacy_detection/extract_features_zeroshot.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "fallacy_detection"
DEFAULT_MODEL = "facebook/bart-large-mnli"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
WORD_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)

# Operational categories inspired by common fallacy taxonomies discussed in
# the reviewed literature. This compact set is for a feasibility experiment;
# it is NOT a verbatim reproduction of MAFALDA/FAINA labels.
FALLACY_LABELS = {
    "ad_hominem": "attacking a person or their character instead of addressing their argument",
    "appeal_to_fear": "using fear or threatening consequences to persuade rather than evidence",
    "appeal_to_authority": "treating an authority's opinion as decisive evidence without adequate support",
    "appeal_to_emotion": "using emotional language or feelings instead of relevant evidence",
    "false_cause": "claiming causation from correlation or insufficient causal evidence",
    "false_dilemma": "presenting only two choices when more alternatives exist",
    "hasty_generalization": "making a broad generalization from too little evidence",
    "slippery_slope": "claiming one step will lead to a chain of extreme consequences without adequate support",
    "straw_man": "misrepresenting an opponent's position to make it easier to attack",
    "whataboutism": "deflecting criticism by pointing to another person's or group's alleged wrongdoing",
    "appeal_to_popularity": "claiming something is true or right because many people believe it",
    "circular_reasoning": "using the conclusion itself as support for the conclusion",
}

def split_sentences(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]

def normalize_columns(df: pd.DataFrame, path: Path) -> tuple[str, str, str | None]:
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    text_col = next((normalized[k] for k in ("text", "content", "articletext", "body") if k in normalized), None)
    title_col = next((normalized[k] for k in ("title", "headline") if k in normalized), None)
    bias_col = normalized.get("bias")
    if text_col is None:
        raise ValueError(f"{path.name}: requires Text/content/article body column")
    if bias_col is None:
        raise ValueError(f"{path.name}: requires Bias column for downstream analysis")
    return text_col, title_col, bias_col

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=int, default=-1, help="-1 = CPU, 0 = first GPU.")
    parser.add_argument("--max-sentences", type=int, default=20,
                        help="Maximum sentences scored per article; use 0 for all sentences.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows per CSV for a smoke test.")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Optional CSV basenames from --input-dir.")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_sentences < 0:
        parser.error("--max-sentences must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit("Install optional dependency: python -m pip install transformers torch") from exc

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    paths = sorted(input_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if args.files:
        wanted = set(args.files)
        paths = [p for p in paths if p.name in wanted]
    if not paths:
        raise SystemExit(f"No matching CSV files found in {input_dir}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier = pipeline(
        task="zero-shot-classification",
        model=args.model,
        device=args.device,
    )
    label_names = list(FALLACY_LABELS.keys())
    label_descriptions = [FALLACY_LABELS[name] for name in label_names]
    manifest = []

    for path in paths:
        print(f"\nProcessing {path.name}")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=args.limit)
        text_col, title_col, bias_col = normalize_columns(df, path)
        titles = df[title_col].fillna("").astype(str).tolist() if title_col else [""] * len(df)
        bodies = df[text_col].fillna("").astype(str).tolist()

        article_sentences: list[list[str]] = []
        all_sentences: list[str] = []
        for body in bodies:
            sentences = split_sentences(body)
            if args.max_sentences:
                sentences = sentences[:args.max_sentences]
            article_sentences.append(sentences)
            all_sentences.extend(sentences)

        # The HF pipeline returns one result per input sentence. Chunking bounds
        # memory use; this remains computationally expensive for full corpora.
        per_sentence: list[dict[str, float]] = []
        for start in range(0, len(all_sentences), args.batch_size):
            batch = all_sentences[start:start + args.batch_size]
            outputs = classifier(
                batch,
                candidate_labels=label_descriptions,
                multi_label=True,
                hypothesis_template="This text illustrates {}.",
            )
            if isinstance(outputs, dict):
                outputs = [outputs]
            for output in outputs:
                score_map = dict(zip(output["labels"], output["scores"]))
                per_sentence.append({
                    label: float(score_map.get(description, 0.0))
                    for label, description in zip(label_names, label_descriptions)
                })

        cursor = 0
        rows = []
        for idx, (title, sentences, body) in enumerate(zip(titles, article_sentences, bodies), start=1):
            scores = per_sentence[cursor:cursor + len(sentences)]
            cursor += len(sentences)
            record: dict[str, Any] = {
                "Article_ID": f"{path.stem}_{idx:06d}",
                "fallacy_zs_sentence_count_scored": len(sentences),
            }
            for label in label_names:
                vals = [s[label] for s in scores]
                record[f"fallacy_zs_{label}_mean_score"] = float(np.mean(vals)) if vals else np.nan
                record[f"fallacy_zs_{label}_max_score"] = float(np.max(vals)) if vals else np.nan
                record[f"fallacy_zs_{label}_sentences_above_0_5"] = int(sum(v >= 0.5 for v in vals))
                record[f"fallacy_zs_{label}_rate_per_100_sentences"] = (
                    100.0 * sum(v >= 0.5 for v in vals) / len(vals) if vals else np.nan
                )
            mean_scores = {
                label: record[f"fallacy_zs_{label}_mean_score"] for label in label_names
            }
            finite = {k: v for k, v in mean_scores.items() if np.isfinite(v)}
            record["fallacy_zs_highest_mean_score_category"] = (
                max(finite, key=finite.get) if finite else ""
            )
            rows.append(record)

        result = pd.DataFrame(rows, index=df.index)  # Article_ID + features only (no article text)
        if bias_col:
            result.insert(1, "Bias", df[bias_col].values)
        out_path = output_dir / f"{path.stem}_fallacy_zeroshot.csv"
        result.to_csv(out_path, index=False)
        manifest.append({
            "dataset": path.name,
            "input_rows": len(df),
            "output_rows": len(result),
            "model": args.model,
            "candidate_categories": "|".join(label_names),
            "max_sentences_per_article": args.max_sentences or "all",
            "output_file": out_path.name,
            "note": "Exploratory zero-shot scores; not validated human fallacy labels.",
        })
        print(f"Saved {len(result):,} rows to {out_path}")

    pd.DataFrame(manifest).to_csv(output_dir / "run_manifest.csv", index=False)

if __name__ == "__main__":
    main()
