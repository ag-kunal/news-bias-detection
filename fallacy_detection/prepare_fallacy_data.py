#!/usr/bin/env python3
"""Build one sentence-level training table from the labelled fallacy corpora in the supplied papers.

Sources (clone them into fallacy_detection/reference_data/, see README):
  MAFALDA   (Helwe et al., NAACL 2024)    github.com/ChadiHelwe/MAFALDA          sentences with fallacy / "nothing"
  ElecDeb   (Goffredo et al., EMNLP 2023) github.com/pierpaologoffredo/FallacyDetection  US debates; fallacy snippets + context
  LOGIC     (Jin et al., 2022)            github.com/causalNLP/logical-fallacy   quiz sentences (edu) and news (climate)

Output columns: text, is_fallacy (1/0/NaN), fallacy_type (harmonised, '' if unknown/none), source,
domain, group (id used to keep sentences of one text in one CV fold), original_label.

is_fallacy is only defined for sources that contain genuine negatives (MAFALDA, ElecDeb contexts).
LOGIC contains positives only, so it is used for the type classifier, never for detection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import OUT_DIR, REF_DIR, n_words, split_sentences

EMOTION = {"appeal to fear", "appeal to anger", "appeal to pity", "appeal to positive emotion", "appeal to ridicule",
           "appeal to emotion"}
OTHER = {"straw man", "false analogy", "equivocation", "appeal to nature", "appeal to tradition",
         "appeal to worse problems", "fallacy of division", "fallacy of extension"}
HARMONISE = {
    "ad hominem": "ad_hominem", "tu quoque": "ad_hominem", "guilt by association": "ad_hominem",
    "appeal to (false) authority": "appeal_to_authority",
    "causal oversimplification": "causal_oversimplification", "false causality": "causal_oversimplification",
    "slippery slope": "slippery_slope", "hasty generalization": "hasty_generalization",
    "faulty generalization": "hasty_generalization", "false dilemma": "false_dilemma",
    "ad populum": "ad_populum", "circular reasoning": "circular_reasoning",
    **{k: "appeal_to_emotion" for k in EMOTION}, **{k: "other_fallacy" for k in OTHER},
}
ELECDEB = {"AdHominem": "ad_hominem", "AppealtoAuthority": "appeal_to_authority",
           "AppealtoEmotion": "appeal_to_emotion", "FalseCause": "causal_oversimplification",
           "Slipperyslope": "slippery_slope"}  # "Slogans" dropped: MAFALDA does not treat slogans as fallacies
# LOGIC umbrella labels (relevance/logic/credibility/intentional/misc) are ambiguous -> no type
LOGIC_UMBRELLA = {"fallacy of relevance", "fallacy of logic", "fallacy of credibility", "intentional", "miscellaneous"}


def norm(label: str) -> str:
    return str(label).strip().lower()


def mafalda(ref: Path) -> list[dict]:
    path = ref / "MAFALDA" / "datasets" / "gold_standard_dataset.jsonl"
    rows = []
    for tid, line in enumerate(path.open(encoding="utf-8")):
        rec = json.loads(line)
        sent_labels = rec["sentences_with_labels"]
        if isinstance(sent_labels, str):
            sent_labels = json.loads(sent_labels)
        for sent, label_sets in sent_labels.items():
            if sent.strip().upper().startswith(("TITLE:", "POST:")) and n_words(sent) < 3:
                continue
            if sent.strip() == "POST:" or n_words(sent) < 4:
                continue
            sets = [[norm(x) for x in s] for s in label_sets]
            obligatory = [s for s in sets if "nothing" not in s]
            if obligatory:
                orig = obligatory[0][0]
                ftype = HARMONISE.get(orig, "")
                if not ftype:  # unmapped annotator label such as a typo
                    continue
                rows.append(dict(text=sent, is_fallacy=1, fallacy_type=ftype, source="mafalda",
                                 domain="reddit/news/debates", group=f"mafalda_{tid}", original_label=orig))
            elif all(s == ["nothing"] for s in sets):
                rows.append(dict(text=sent, is_fallacy=0, fallacy_type="", source="mafalda",
                                 domain="reddit/news/debates", group=f"mafalda_{tid}", original_label="nothing"))
            # else: only optional alternatives -> ambiguous, left out of training
    return rows


def elecdeb(ref: Path) -> list[dict]:
    base = ref / "FallacyDetection" / "data"
    frames = [pd.read_csv(base / f) for f in ("pos_train_set.csv", "pos_test_set.csv") if (base / f).exists()]
    df = pd.concat(frames, ignore_index=True)
    snippets = [str(s) for s in df["Fallacy"]]
    snippet_set = set(snippets)
    rows = []
    for i, r in df.iterrows():
        ftype = ELECDEB.get(r["Label"], "")
        if not ftype:
            continue
        rows.append(dict(text=str(r["Fallacy"]), is_fallacy=1, fallacy_type=ftype, source="elecdeb",
                         domain="us_debates", group=f"elecdeb_{i}", original_label=r["Label"]))
        # Negatives: neighbouring context sentences that are not (part of) any annotated fallacy.
        ctx = str(r["Context"]).replace(str(r["Fallacy"]), " ")
        for s in split_sentences(ctx):
            if n_words(s) < 6 or s in snippet_set:
                continue
            if any((s in sn) or (sn in s) for sn in snippets if abs(len(sn) - len(s)) < 400):
                continue
            rows.append(dict(text=s, is_fallacy=0, fallacy_type="", source="elecdeb", domain="us_debates",
                             group=f"elecdeb_{i}", original_label="context_not_annotated"))
    return rows


def logic(ref: Path) -> list[dict]:
    base = ref / "logical-fallacy" / "data"
    rows = []
    for fname, col, source, domain in (("edu_all.csv", "updated_label", "logic_edu", "quiz"),
                                       ("climate_all.csv", "logical_fallacies", "logic_climate", "climate_news")):
        df = pd.read_csv(base / fname)
        for i, r in df.iterrows():
            orig = norm(r[col])
            if orig == "miscellaneous":
                continue
            ftype = "" if orig in LOGIC_UMBRELLA else HARMONISE.get(orig, "")
            rows.append(dict(text=str(r["source_article"]), is_fallacy=float("nan"), fallacy_type=ftype,
                             source=source, domain=domain, group=f"{source}_{i}", original_label=orig))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", type=Path, default=REF_DIR)
    ap.add_argument("--output", type=Path, default=OUT_DIR / "training" / "fallacy_sentences.csv")
    args = ap.parse_args()
    rows = []
    for name, fn in (("MAFALDA", mafalda), ("ElecDeb", elecdeb), ("LOGIC", logic)):
        try:
            got = fn(args.ref_dir)
            print(f"{name}: {len(got):,} rows")
            rows += got
        except FileNotFoundError as exc:
            print(f"SKIP {name}: {exc}")
    if not rows:
        raise SystemExit(f"No reference data found under {args.ref_dir}. See fallacy_detection/README.md.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["text", "source"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} rows -> {args.output}")
    print(df.groupby(["source", "is_fallacy"], dropna=False).size().to_string())
    print("\nType counts (positives with a harmonised type):")
    print(df[df.fallacy_type != ""].groupby(["fallacy_type", "source"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
