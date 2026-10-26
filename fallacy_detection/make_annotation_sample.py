#!/usr/bin/env python3
"""Draw a blind sample of news sentences for manual fallacy annotation (feasibility check).

The sample is stratified by Bias label and written WITHOUT model scores so annotators are not primed.
A second file keeps the scores (keyed by sample_id) for comparison after annotation. Two annotators should label
each sentence independently: is_fallacy (1/0), type (one of the working types or 'other'), then compare against the model
with precision/recall and report inter-annotator agreement (Cohen's kappa) as in MAFALDA/FAINA.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import pandas as pd

from common import OUT_DIR, ROOT, TYPES, split_sentences


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="One dataset CSV (e.g. data/Combined_Dataset.csv).")
    ap.add_argument("--model", type=Path, default=OUT_DIR / "model" / "fallacy_models.joblib")
    ap.add_argument("--per-class", type=int, default=100, help="Sentences per Bias label.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR / "annotation")
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    text_col = next(cols[k] for k in ("text", "content", "articletext", "body") if k in cols)
    bias_col = cols["bias"]
    rows = []
    for i, r in df.iterrows():
        for s in split_sentences(r[text_col]):
            if len(re.findall(r"\w+", s)) >= 8:
                rows.append({"article_row": i + 1, "Bias": str(r[bias_col]).strip().lower(), "sentence": s})
    pool = pd.DataFrame(rows)
    parts = [g.sample(min(len(g), args.per_class), random_state=args.seed) for _, g in pool.groupby("Bias")]
    sample = pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sample.insert(0, "sample_id", range(1, len(sample) + 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blank = sample[["sample_id", "sentence"]].copy()
    for a in ("annotator1", "annotator2"):
        blank[f"{a}_is_fallacy"] = ""
        blank[f"{a}_type"] = ""
    blank.to_csv(args.output_dir / "annotation_sample_blind.csv", index=False)
    meta = sample[["sample_id", "article_row", "Bias"]].copy()
    if args.model.exists():
        b = joblib.load(args.model)
        meta["model_p_fallacy"] = b["detector"].predict_proba(sample["sentence"])[:, 1]
        meta["model_type"] = [b["typer"].classes_[k] for k in b["typer"].predict_proba(sample["sentence"]).argmax(1)]
    meta.to_csv(args.output_dir / "annotation_sample_model_scores.csv", index=False)
    print(f"Wrote {len(sample)} sentences. Allowed types: {', '.join(TYPES)}")


if __name__ == "__main__":
    main()
