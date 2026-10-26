#!/usr/bin/env python3
"""Stage 2: article-level fallacy features from sentence-level fallacy models.

Applies the detector/type models trained by train_fallacy_classifier.py to every sentence of every article and
aggregates per article. Output: outputs/fallacy_detection/<dataset>_fallacy_features.csv with Article_ID, Bias
(if present) and the feature columns only (no article text), so it merges with the other modules on Article_ID.

Features per article (n = scored sentences):
  fallacy_sentence_count_scored
  fallacy_p_mean, fallacy_p_max           mean / max sentence P(fallacy)
  fallacy_flagged_rate                    share of sentences with P(fallacy) >= tuned threshold
  fallacy_<type>_soft_share               mean over sentences of P(fallacy) * P(type | fallacy)
  fallacy_<type>_flagged_share            share of sentences flagged AND with this type as arg-max

These are WEAK model-generated signals, not human annotations. The detector does not transfer well across
corpora (see outputs/fallacy_detection/training/detector_metrics.csv), so validate on a hand-labelled news sample
(make_annotation_sample.py) before using them as paper evidence.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import OUT_DIR, ROOT, split_sentences

DEFAULT_INPUT_DIR = ROOT / "data"


def pick(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    return next((norm[n] for n in names if n in norm), None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--model", type=Path, default=OUT_DIR / "model" / "fallacy_models.joblib")
    ap.add_argument("--files", nargs="*", default=None, help="CSV basenames in --input-dir to process.")
    ap.add_argument("--limit", type=int, default=None, help="First N rows per CSV (smoke test).")
    ap.add_argument("--min-words", type=int, default=5, help="Skip sentences shorter than this.")
    ap.add_argument("--max-sentences", type=int, default=0, help="Max sentences per article (0 = all).")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run prepare_fallacy_data.py then train_fallacy_classifier.py.")
    bundle = joblib.load(args.model)
    detector, typer, thr, types = bundle["detector"], bundle["typer"], bundle["threshold"], list(bundle["types"])

    paths = sorted(args.input_dir.expanduser().resolve().glob("*.csv"), key=lambda p: p.name.lower())
    if args.files:
        paths = [p for p in paths if p.name in set(args.files)]
    if not paths:
        raise SystemExit(f"No matching CSV files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for path in paths:
        print(f"Processing {path.name}")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=args.limit)
        text_col = pick(df, ("text", "content", "articletext", "body"))
        if text_col is None:
            print(f"  SKIP: no text column"); continue
        bias_col = pick(df, ("bias",))
        id_col = pick(df, ("articleid",))
        ids = df[id_col].astype(str).tolist() if id_col else [f"{path.stem}_{i:06d}" for i in range(1, len(df) + 1)]

        owners, sents = [], []
        for i, body in enumerate(df[text_col]):
            ss = [s for s in split_sentences(body) if len(re.findall(r"\w+", s)) >= args.min_words]
            if args.max_sentences:
                ss = ss[: args.max_sentences]
            owners += [i] * len(ss); sents += ss
        owners = np.asarray(owners, dtype=int)
        if sents:
            p_f = detector.predict_proba(sents)[:, 1]
            p_t = typer.predict_proba(sents)  # columns follow typer.classes_
            order = [list(typer.classes_).index(t) for t in types]
            p_t = p_t[:, order]
        else:
            p_f, p_t = np.zeros(0), np.zeros((0, len(types)))
        flagged = p_f >= thr
        argmax = p_t.argmax(1) if len(p_t) else np.zeros(0, dtype=int)

        n = len(df)
        feats = {"fallacy_sentence_count_scored": np.bincount(owners, minlength=n) if len(owners) else np.zeros(n, int)}
        cnt = feats["fallacy_sentence_count_scored"].astype(float)
        safe = np.where(cnt > 0, cnt, np.nan)

        def agg_sum(values):
            return np.bincount(owners, weights=values, minlength=n) if len(owners) else np.zeros(n)

        feats["fallacy_p_mean"] = agg_sum(p_f) / safe
        pmax = np.full(n, np.nan)
        for i in np.unique(owners):
            pmax[i] = p_f[owners == i].max()
        feats["fallacy_p_max"] = pmax
        feats["fallacy_flagged_rate"] = agg_sum(flagged.astype(float)) / safe
        for j, t in enumerate(types):
            feats[f"fallacy_{t}_soft_share"] = agg_sum(p_f * p_t[:, j]) / safe if len(owners) else np.full(n, np.nan)
            feats[f"fallacy_{t}_flagged_share"] = agg_sum((flagged & (argmax == j)).astype(float)) / safe
        out = pd.DataFrame(feats)
        out.insert(0, "Article_ID", ids)
        if bias_col:
            out.insert(1, "Bias", df[bias_col].values)
        out_path = args.output_dir / f"{path.stem}_fallacy_features.csv"
        out.to_csv(out_path, index=False)
        manifest.append({"dataset": path.name, "input_rows": n, "output_rows": len(out), "sentences_scored": len(sents),
                         "articles_without_scored_sentences": int((cnt == 0).sum()), "flag_threshold": thr,
                         "overall_flagged_rate": float(flagged.mean()) if len(flagged) else np.nan,
                         "output_file": out_path.name,
                         "note": "Weak model-generated signal; not human fallacy annotation."})
        print(f"  saved {len(out):,} rows; {len(sents):,} sentences; flagged rate {manifest[-1]['overall_flagged_rate']:.2%}")
    pd.DataFrame(manifest).to_csv(args.output_dir / "run_manifest.csv", index=False)


if __name__ == "__main__":
    main()
