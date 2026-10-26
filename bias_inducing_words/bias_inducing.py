#!/usr/bin/env python3
"""Extract title-to-sentence semantic congruence features (paper Section 3.1.3).

The paper does not specify an embedding checkpoint. The default checkpoint is an
explicit implementation choice, recorded in the manifest. This is topic-congruence
measurement, not a validated list of biased words or a direct bias label.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "outputs" / "bias_inducing_words"
SENT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")

def split_sentences(text):
    text = "" if pd.isna(text) else str(text).strip()
    return [s.strip() for s in SENT_RE.split(text) if s.strip()]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--files", nargs="*", default=None)
    args = ap.parse_args()
    if args.batch_size < 1: ap.error("--batch-size must be >= 1")
    if args.limit is not None and args.limit < 1: ap.error("--limit must be >= 1")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("Install optional dependency: python -m pip install sentence-transformers") from exc
    input_dir = args.input_dir.expanduser().resolve()
    paths = sorted(input_dir.glob("*.csv"), key=lambda p: p.name.lower())
    if args.files:
        wanted = set(args.files)
        paths = [p for p in paths if p.name in wanted]
    if not paths:
        raise SystemExit(f"No matching CSV files found in {input_dir}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model)
    manifest = []

    for path in paths:
        print(f"Processing {path.name}")
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=args.limit)
        norm = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
        text_col = next((norm[k] for k in ("text", "content", "articletext", "body") if k in norm), None)
        title_col = next((norm[k] for k in ("title", "headline") if k in norm), None)
        if text_col is None or title_col is None:
            print(f"SKIP {path.name}: title/headline and text/content required")
            continue
        titles = df[title_col].fillna("").astype(str).tolist()
        sentence_lists = [split_sentences(x) for x in df[text_col].fillna("").astype(str)]
        # Encode all titles once; sentence embeddings are processed in bounded batches
        # so large datasets do not require storing every sentence string/embedding at once.
        title_inputs = [t if t.strip() else " " for t in titles]
        title_embeddings = model.encode(
            title_inputs, batch_size=args.batch_size, show_progress_bar=True,
            normalize_embeddings=True, convert_to_numpy=True
        )
        records = []
        for start in range(0, len(df), 128):
            end = min(start + 128, len(df))
            batch_sents = []
            owners = []
            for local_i in range(start, end):
                for sent in sentence_lists[local_i]:
                    batch_sents.append(sent)
                    owners.append(local_i)
            if batch_sents:
                sent_embeddings = model.encode(
                    batch_sents, batch_size=args.batch_size, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True
                )
                sims_by_article = {i: [] for i in range(start, end)}
                for owner, embedding in zip(owners, sent_embeddings):
                    # A blank title has no meaning to compare against: leave it undefined.
                    if titles[owner].strip():
                        sims_by_article[owner].append(float(np.dot(title_embeddings[owner], embedding)))
            else:
                sims_by_article = {i: [] for i in range(start, end)}
            for i in range(start, end):
                sims = np.asarray(sims_by_article[i], dtype=float)
                records.append({
                    "title_sentence_count": len(sentence_lists[i]) if titles[i].strip() else 0,
                    "title_content_cosine_similarity_mean": float(np.mean(sims)) if len(sims) else np.nan,
                    "title_content_cosine_similarity_median": float(np.median(sims)) if len(sims) else np.nan,
                    "title_content_cosine_similarity_min": float(np.min(sims)) if len(sims) else np.nan,
                    "title_content_cosine_similarity_max": float(np.max(sims)) if len(sims) else np.nan,
                })
            print(f"  processed {end:,}/{len(df):,} articles")
        result = df.copy()
        if "Article_ID" not in result.columns:
            result.insert(0, "Article_ID", [f"{path.stem}_{i:06d}" for i in range(1, len(df) + 1)])
        result = pd.concat([result, pd.DataFrame(records, index=df.index)], axis=1)
        out = output_dir / f"{path.stem}_topic_congruence.csv"
        result.to_csv(out, index=False)
        manifest.append({
            "dataset": path.name, "input_rows": len(df), "output_rows": len(result),
            "output_file": out.name, "model": args.model,
            "method": "mean/median/min/max cosine similarity between title embedding and sentence embeddings",
            "thresholds": "none",
            "caveat": "Model checkpoint is an implementation choice; similarity is not a direct bias label."
        })
        print(f"  saved {out}")
    pd.DataFrame(manifest).to_csv(output_dir / "run_manifest.csv", index=False)

if __name__ == "__main__":
    main()
