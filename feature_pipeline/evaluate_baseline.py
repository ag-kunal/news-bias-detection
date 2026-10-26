#!/usr/bin/env python3
"""Leakage-aware baseline evaluation on merged feature tables, with feature-group ablation.

What it does
* Repeated-free, grouped, stratified K-fold cross-validation (StratifiedGroupKFold). Articles with
  the same normalised title (or the same Source, with --group-by source) never appear in both train
  and test, which blocks near-duplicate and publisher leakage. Preprocessing (imputation, scaling)
  lives inside the sklearn Pipeline, so it is fitted on training folds only.
* Reports macro-F1 (mean +- std over folds) and, with --binary, the F1 of the "biased" class
  (left/right vs center), which is the metric style used in the draft's Tables 1-4.
* Evaluates each feature group alone and cumulatively (e.g. Stage 1 only vs Stage 1 + fallacy),
  using outputs/combined/feature_groups.json written by merge_features.py.
* Always reports a majority-class baseline.

Source/publisher identity is never used as a feature unless --include-source is passed.
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ["sentiment", "adjective", "punctuation", "readability", "topic_congruence"]
META = {"article_id", "title", "headline", "text", "content", "articletext", "body", "bias",
        "source", "url", "date", "author", "storyid"}


def canon(x: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def make_model(name: str, numeric: list[str], categorical: list[str]) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    pre = ColumnTransformer(transformers, remainder="drop")
    if name == "lr":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    elif name == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced")
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])


def cv_scores(df, y, cols, groups, model_name, k, seed, positive=None):
    X = df[cols]
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    macro, acc, dummy_macro, pos_f1 = [], [], [], []
    for tr, te in cv.split(X, y, groups):
        m = make_model(model_name, cols, [])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X.iloc[tr], y.iloc[tr])
            pred = m.predict(X.iloc[te])
        d = DummyClassifier(strategy="most_frequent").fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        macro.append(f1_score(y.iloc[te], pred, average="macro", zero_division=0))
        acc.append(accuracy_score(y.iloc[te], pred))
        dummy_macro.append(f1_score(y.iloc[te], d, average="macro", zero_division=0))
        if positive is not None:
            pos_f1.append(f1_score(y.iloc[te], pred, pos_label=positive, zero_division=0))
    out = {"macro_f1_mean": np.mean(macro), "macro_f1_std": np.std(macro), "accuracy_mean": np.mean(acc),
           "majority_macro_f1_mean": np.mean(dummy_macro)}
    if positive is not None:
        out["biased_class_f1_mean"] = np.mean(pos_f1)
        out["biased_class_f1_std"] = np.std(pos_f1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=ROOT / "outputs" / "combined")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "evaluation")
    ap.add_argument("--files", nargs="*", default=None, help="Dataset stems to evaluate, e.g. Combined_Dataset")
    ap.add_argument("--models", nargs="+", default=["lr"], choices=["lr", "svm", "rf"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--group-by", choices=["title", "source", "none"], default="title",
                    help="Keep rows sharing this key in the same fold (default: normalised title).")
    ap.add_argument("--binary", action="store_true",
                    help="Collapse labels to biased (left/right) vs unbiased (center) and report biased-class F1.")
    ap.add_argument("--include-source", action="store_true", help="Use Source as a feature (leaks publisher identity).")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_dir.glob("*_combined_features.csv"))
    if args.files:
        paths = [p for p in paths if p.name.replace("_combined_features.csv", "") in set(args.files)]
    if not paths:
        raise SystemExit(f"No *_combined_features.csv files in {args.input_dir}; run merge_features.py first.")
    groups_file = args.input_dir / "feature_groups.json"
    all_groups = json.loads(groups_file.read_text()) if groups_file.exists() else {}

    rows = []
    for path in paths:
        stem = path.name.replace("_combined_features.csv", "")
        df = pd.read_csv(path, low_memory=False)
        bias_col = next((c for c in df.columns if c.lower() == "bias"), None)
        if bias_col is None:
            print(f"SKIP {stem}: no Bias column"); continue
        y = df[bias_col].astype("string").str.strip().str.lower()
        ok = y.notna() & (y != "")
        df, y = df.loc[ok].reset_index(drop=True), y.loc[ok].astype(str).reset_index(drop=True)
        positive = None
        if args.binary:
            y = y.map(lambda v: "unbiased" if v == "center" else "biased")
            positive = "biased"
        if y.nunique() < 2:
            print(f"SKIP {stem}: fewer than 2 classes"); continue

        if args.group_by == "title" and any(canon(c) in ("title", "headline") for c in df.columns):
            tcol = next(c for c in df.columns if canon(c) in ("title", "headline"))
            groups = df[tcol].fillna("").astype(str).map(canon)
            groups = groups.where(groups != "", other=pd.Series(df.index.astype(str), index=df.index))
        elif args.group_by == "source" and "Source" in df.columns:
            groups = df["Source"].astype(str)
        else:
            groups = pd.Series(np.arange(len(df)))
        groups = pd.factorize(groups)[0]

        mod_cols = all_groups.get(stem, {})
        if not mod_cols:  # fall back to "all numeric columns" as one group
            num = [c for c in df.columns if canon(c) not in META and pd.api.types.is_numeric_dtype(df[c])]
            mod_cols = {"all_numeric": num}
        clean = {}
        for mod, cols in mod_cols.items():
            keep = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
                    and canon(c) not in META and df[c].notna().any() and df[c].nunique(dropna=True) > 1]
            if keep:
                clean[mod] = keep
        stage1 = [m for m in STAGE1 if m in clean]
        experiments = {f"{m} only": clean[m] for m in clean}
        if stage1:
            experiments["STAGE1 (all existing modules)"] = sum((clean[m] for m in stage1), [])
        if "fallacy" in clean and stage1:
            experiments["STAGE1 + fallacy"] = experiments["STAGE1 (all existing modules)"] + clean["fallacy"]
        experiments["ALL modules"] = sum(clean.values(), [])

        k = min(args.folds, int(y.value_counts().min()))
        if k < 2:
            print(f"SKIP {stem}: a class has fewer than 2 rows"); continue
        for exp, cols in experiments.items():
            cols = list(dict.fromkeys(cols))
            for model_name in args.models:
                res = cv_scores(df, y, cols, groups, model_name, k, args.random_state, positive)
                rows.append({"dataset": stem, "experiment": exp, "model": model_name, "n_rows": len(df),
                             "n_features": len(cols), "folds": k, "group_by": args.group_by,
                             "label_mode": "binary" if args.binary else "3-class", **res})
                print(f"{stem:22s} {exp:32s} {model_name:3s} macro-F1={res['macro_f1_mean']:.3f}"
                      f"+-{res['macro_f1_std']:.3f} (majority {res['majority_macro_f1_mean']:.3f})")
    if not rows:
        raise SystemExit("Nothing could be evaluated.")
    out = pd.DataFrame(rows)
    tag = "binary" if args.binary else "3class"
    out.to_csv(args.output_dir / f"baseline_metrics_{tag}.csv", index=False)
    print(f"Saved {args.output_dir / f'baseline_metrics_{tag}.csv'}")
    print("Note: use --group-by source for the strictest test; Bias labels are usually publisher-level.")


if __name__ == "__main__":
    main()
