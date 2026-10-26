#!/usr/bin/env python3
"""Merge per-module feature CSVs by Article_ID, never by row position.

Reads the row-level feature files written by each extractor under outputs/<module>/...
and writes one table per dataset to outputs/combined/.

Design notes
* Feature columns = columns of a feature file that are NOT in the raw source CSV and are
  not article metadata (title/text/bias/source...). Raw source columns are never re-added.
* If two modules produce a column with the same name (e.g. word_count from both the
  punctuation and readability modules, which tokenize differently) the second one is kept
  under a module prefix (readability__word_count) instead of being silently dropped.
* The merge is a left join on Article_ID. Coverage per feature file is reported, because a
  file produced with --limit covers only the first N rows and would otherwise leave NaNs.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUTS = ROOT / "outputs"

# file-name token -> module label (used for prefixes and the manifest)
MODULE_TOKENS = {
    "punctuation": "punctuation",
    "sentiment": "sentiment",
    "readability": "readability",
    "adjectives": "adjective",
    "topic_congruence": "topic_congruence",
    "fallacy_features": "fallacy",
    "fallacy_zeroshot": "fallacy_zs",
}
METADATA = {"title", "headline", "text", "content", "articletext", "body", "bias", "source",
            "url", "date", "author", "articleid", "storyid"}


def canon(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    norm = {canon(c): c for c in df.columns}
    text_col = next((norm[k] for k in ("text", "content", "articletext", "body") if k in norm), None)
    if text_col is None or "bias" not in norm:
        raise ValueError(f"{path.name}: requires text and Bias columns; found {list(df.columns)}")
    id_col = norm.get("articleid")
    if id_col is not None and id_col != "Article_ID":
        df = df.rename(columns={id_col: "Article_ID"})
    if "Article_ID" not in df.columns:
        df.insert(0, "Article_ID", [f"{path.stem}_{i:06d}" for i in range(1, len(df) + 1)])
    if df["Article_ID"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate Article_ID values")
    return df


def feature_files(outputs: Path, dataset_stem: str) -> list[tuple[str, Path]]:
    found = []
    for path in outputs.rglob("*.csv"):
        name = path.name.lower()
        if not path.name.startswith(dataset_stem + "_"):
            continue
        if any(part.lower() in {"summaries", "summary", "plots", "combined", "evaluation", "results"}
               for part in path.parts):
            continue
        for token, module in MODULE_TOKENS.items():
            if name == f"{dataset_stem.lower()}_{token}.csv":
                found.append((module, path))
                break
    return sorted(set(found), key=lambda x: (x[0], str(x[1])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS / "combined")
    args = ap.parse_args()

    sources = sorted(args.data_dir.glob("*.csv"))
    if not sources:
        raise SystemExit(f"No source CSV files found in {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    groups: dict[str, dict[str, list[str]]] = {}

    for source in sources:
        base = read_source(source)
        raw_cols = set(base.columns)
        merged = base.copy()
        original_rows = len(merged)
        added: list[str] = []
        used_modules: list[str] = []
        coverage: list[str] = []

        for module, fpath in feature_files(args.outputs_dir, source.stem):
            feat = pd.read_csv(fpath, encoding="utf-8-sig", low_memory=False)
            if "Article_ID" not in feat.columns:
                print(f"SKIP {fpath}: no Article_ID; cannot merge safely")
                continue
            if feat["Article_ID"].duplicated().any():
                print(f"SKIP {fpath}: duplicate Article_ID")
                continue
            cols = [c for c in feat.columns
                    if c != "Article_ID" and c not in raw_cols and canon(c) not in METADATA]
            rename = {}
            for c in cols:
                if c in merged.columns:
                    rename[c] = f"{module}__{c}"
            if not cols:
                continue
            sub = feat[["Article_ID"] + cols].rename(columns=rename)
            before = len(merged)
            merged = merged.merge(sub, on="Article_ID", how="left", validate="one_to_one")
            if len(merged) != before:
                raise RuntimeError(f"Merge changed row count for {source.name} via {fpath.name}")
            share = feat["Article_ID"].isin(base["Article_ID"]).sum() / max(len(base), 1)
            coverage.append(f"{module}:{share:.0%}")
            if share < 1.0:
                print(f"WARNING {fpath.name}: covers {share:.1%} of {source.name} rows "
                      f"(was it run with --limit?). Remaining rows will be NaN.")
            if rename:
                print(f"  {module}: renamed duplicate columns -> {sorted(rename.values())}")
            added.extend(rename.get(c, c) for c in cols)
            groups.setdefault(source.stem, {}).setdefault(module, []).extend(rename.get(c, c) for c in cols)
            used_modules.append(module)

        out = args.output_dir / f"{source.stem}_combined_features.csv"
        merged.to_csv(out, index=False)
        manifest.append({
            "dataset": source.name, "source_rows": original_rows, "output_rows": len(merged),
            "modules_merged": "|".join(used_modules), "coverage": "|".join(coverage),
            "feature_columns_added": len(added), "output_file": out.name,
            "row_count_preserved": original_rows == len(merged),
        })
        print(f"{source.name}: {original_rows:,} rows; modules={used_modules}; "
              f"{len(added)} feature columns -> {out}")

    pd.DataFrame(manifest).to_csv(args.output_dir / "merge_manifest.csv", index=False)
    (args.output_dir / "feature_groups.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    print(f"Manifest: {args.output_dir / 'merge_manifest.csv'}")


if __name__ == "__main__":
    main()
