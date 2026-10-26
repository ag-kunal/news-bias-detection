#!/usr/bin/env python3
"""Train and honestly evaluate the Stage 2 fallacy models (TF-IDF + logistic regression).

Two models are trained from outputs/fallacy_detection/training/fallacy_sentences.csv:
  detector   sentence -> P(fallacy). Trained only on sources that have real negatives (MAFALDA, ElecDeb).
  typer      sentence -> P(type | fallacy). Trained on every labelled positive with a harmonised type.

Evaluation answers the feasibility question, "does this transfer?", so besides grouped K-fold it
reports leave-one-source-out results (train on one corpus, test on another). Expect these to be
much lower than in-corpus numbers; the papers report the same drop for zero-shot and fine-tuned models.

This is a deliberately light, CPU-only baseline. The papers use fine-tuned transformers
(FAINA: AlBERTo/UmBERTo; Cantin & Chust: RoBERTa; Goffredo: BERT). A transformer encoder can be swapped in
later behind the same predict_proba interface.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from common import OUT_DIR, TYPES

DETECTOR_SOURCES = ["mafalda", "elecdeb"]


def make_pipeline(C: float = 2.0) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, lowercase=True)),
    ])
    return Pipeline([("tfidf", feats),
                     ("lr", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


def best_threshold(y, p) -> float:
    grid = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def det_metrics(y, p, thr):
    pred = p >= thr
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    return dict(precision=pr, recall=rc, f1_fallacy=f1, macro_f1=f1_score(y, pred, average="macro", zero_division=0),
                roc_auc=auc, positive_rate=float(np.mean(y)), n=len(y))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=OUT_DIR / "training" / "fallacy_sentences.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.data.exists():
        raise SystemExit(f"{args.data} not found. Run prepare_fallacy_data.py first.")
    df = pd.read_csv(args.data, keep_default_na=False, na_values=[""])
    df["fallacy_type"] = df["fallacy_type"].fillna("")
    tr_dir = args.out_dir / "training"; model_dir = args.out_dir / "model"
    tr_dir.mkdir(parents=True, exist_ok=True); model_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    report: dict = {}

    # ---------------- detector ----------------
    det = df[df["source"].isin(DETECTOR_SOURCES) & df["is_fallacy"].notna()].reset_index(drop=True)
    y = det["is_fallacy"].astype(int).to_numpy()
    rows = []
    # grouped CV on the pooled data
    oof = np.zeros(len(det))
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for tr, te in cv.split(det["text"], y, det["group"]):
        m = make_pipeline().fit(det["text"].iloc[tr], y[tr])
        oof[te] = m.predict_proba(det["text"].iloc[te])[:, 1]
    thr = best_threshold(y, oof)  # tuned on out-of-fold predictions, not on a held-out test set
    rows.append({"setting": f"pooled grouped {args.folds}-fold CV", "train": "mafalda+elecdeb", "test": "mafalda+elecdeb",
                 "threshold": thr, **det_metrics(y, oof, thr)})
    for src in DETECTOR_SOURCES:
        sub = det["source"] == src
        rows.append({"setting": "pooled CV, per-source slice", "train": "mafalda+elecdeb", "test": src,
                     "threshold": thr, **det_metrics(y[sub], oof[sub], thr)})
    # leave-one-source-out
    for src in DETECTOR_SOURCES:
        tr_mask = det["source"] != src
        m = make_pipeline().fit(det.loc[tr_mask, "text"], y[tr_mask])
        p = m.predict_proba(det.loc[~tr_mask, "text"])[:, 1]
        rows.append({"setting": "leave-one-source-out (transfer)", "train": "other", "test": src,
                     "threshold": thr, **det_metrics(y[~tr_mask], p, thr)})
    pd.DataFrame(rows).to_csv(tr_dir / "detector_metrics.csv", index=False)
    print("DETECTOR (fallacy vs none)")
    print(pd.DataFrame(rows)[["setting", "train", "test", "precision", "recall", "f1_fallacy", "roc_auc", "n"]]
          .round(3).to_string(index=False))

    detector = make_pipeline().fit(det["text"], y)

    # ---------------- type classifier ----------------
    typ = df[df["fallacy_type"].isin(TYPES)].reset_index(drop=True)
    ty = typ["fallacy_type"].to_numpy()
    trows = []
    oof_t = np.empty(len(typ), dtype=object)
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for tr, te in cv.split(typ["text"], ty, typ["group"]):
        m = make_pipeline().fit(typ["text"].iloc[tr], ty[tr])
        oof_t[te] = m.predict(typ["text"].iloc[te])
    trows.append({"setting": "pooled grouped CV", "test": "all", "macro_f1": f1_score(ty, oof_t, average="macro"),
                  "accuracy": float(np.mean(ty == oof_t)), "n": len(typ)})
    for src in typ["source"].unique():
        mask = typ["source"] == src
        m = make_pipeline().fit(typ.loc[~mask, "text"], ty[~mask])
        pred = m.predict(typ.loc[mask, "text"])
        trows.append({"setting": "leave-one-source-out (transfer)", "test": src,
                      "macro_f1": f1_score(ty[mask], pred, average="macro", labels=sorted(set(ty[mask])), zero_division=0),
                      "accuracy": float(np.mean(ty[mask] == pred)), "n": int(mask.sum())})
    per_class = precision_recall_fscore_support(ty, oof_t, labels=TYPES, zero_division=0)
    pc = pd.DataFrame({"type": TYPES, "precision": per_class[0], "recall": per_class[1], "f1": per_class[2],
                       "support": per_class[3]})
    pd.DataFrame(trows).to_csv(tr_dir / "type_metrics.csv", index=False)
    pc.to_csv(tr_dir / "type_per_class_pooled_cv.csv", index=False)
    print("\nTYPE CLASSIFIER (given fallacy)")
    print(pd.DataFrame(trows).round(3).to_string(index=False))
    print(pc.round(3).to_string(index=False))
    typer = make_pipeline().fit(typ["text"], ty)

    joblib.dump({"detector": detector, "threshold": thr, "typer": typer, "types": list(typer.classes_),
                 "trained_on": {"detector": DETECTOR_SOURCES, "typer": sorted(typ["source"].unique())},
                 "n_detector": len(det), "n_typer": len(typ)}, model_dir / "fallacy_models.joblib")
    (tr_dir / "training_summary.json").write_text(json.dumps(
        {"detector_threshold": thr, "n_detector_rows": len(det), "n_type_rows": len(typ),
         "caveat": "Light TF-IDF baseline; transfer scores (leave-one-source-out) are the realistic estimate."}, indent=2))
    print(f"\nSaved model -> {model_dir / 'fallacy_models.joblib'} (detector threshold {thr:.2f})")


if __name__ == "__main__":
    main()
