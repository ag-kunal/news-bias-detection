"""Shared constants and helpers for the Stage 2 fallacy modules."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF_DIR = Path(__file__).resolve().parent / "reference_data"
OUT_DIR = ROOT / "outputs" / "fallacy_detection"

# Working taxonomy: the labels shared by (at least) two of the supervised corpora.
# Not a verbatim copy of MAFALDA, FAINA or ElecDeb; see README for the mapping.
TYPES = [
    "ad_hominem", "appeal_to_authority", "appeal_to_emotion", "causal_oversimplification",
    "slippery_slope", "hasty_generalization", "false_dilemma", "ad_populum",
    "circular_reasoning", "other_fallacy",
]

SENT_RE = re.compile(r"(?<=[.!?])\s+|[\r\n]+")


def split_sentences(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [s.strip() for s in SENT_RE.split(str(value)) if s.strip()]


def n_words(s: str) -> int:
    return len(re.findall(r"\w+", s))
