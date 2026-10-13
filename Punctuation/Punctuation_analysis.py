
import re
import unicodedata
from collections import Counter

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import sent_tokenize


# Download the NLTK sentence-tokenization resources.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

sns.set_theme(style="whitegrid")

DATASET_PATH = "Combined_Dataset.csv"
OUTPUT_CSV = "punctuation_analysis_results.csv"
PLOT_DIR = "punctuation_plots"


# ---------------------------------------------------------
# Common helper functions
# ---------------------------------------------------------

def count_words(text):
    """Count word-like tokens without counting punctuation."""
    if not isinstance(text, str) or not text.strip():
        return 0

    return len(re.findall(r"\b[\w]+(?:['’][\w]+)*\b", text))


def count_sentences(text):
    """Count sentences using NLTK."""
    if not isinstance(text, str) or not text.strip():
        return 0

    return len(sent_tokenize(text.strip()))


def count_all_punctuation(text):
    """Count Unicode punctuation characters in the text."""
    if not isinstance(text, str):
        return 0

    return sum(
        1
        for char in text
        if unicodedata.category(char).startswith("P")
    )


def count_nonterminal_punctuation(text):
    """
    Count punctuation excluding periods, question marks,
    and exclamation marks.
    """
    if not isinstance(text, str):
        return 0

    excluded = {".", "?", "!"}

    return sum(
        1
        for char in text
        if (
            unicodedata.category(char).startswith("P")
            and char not in excluded
        )
    )


# Punctuation categories used in features 3 and 4.
PUNCTUATION_PATTERNS = {
    "comma": r",",
    "semicolon": r";",
    "colon": r":",
    "dash": r"[-\u2010-\u2015\u2212]",
    "parentheses": r"[()]",
    "quotation_marks": r"""["'\u2018\u2019\u201c\u201d\u00ab\u00bb]""",
    "question_mark": r"\?",
    "exclamation_mark": r"!",
    "period": r"\.",
    "square_brackets": r"[\[\]]",
    "slash": r"[/\\]",
    "ellipsis_character": r"\u2026",
}


def get_punctuation_counts(text):
    """Return counts for each selected punctuation category."""
    if not isinstance(text, str):
        text = ""

    return {
        name: len(re.findall(pattern, text))
        for name, pattern in PUNCTUATION_PATTERNS.items()
    }


# ---------------------------------------------------------
# FEATURE 1: Punctuation density per 100 words
# ---------------------------------------------------------

def punctuation_density(text):
    """
    Return punctuation density per 100 words.

    Returns both:
      - density_all: all punctuation
      - density_nonterminal: excluding '.', '?' and '!'

    Returns NaN when the text contains no words.
    """
    words = count_words(text)

    if words == 0:
        return {
            "punctuation_density_all": float("nan"),
            "punctuation_density_nonterminal": float("nan"),
        }

    total = count_all_punctuation(text)
    nonterminal = count_nonterminal_punctuation(text)

    return {
        "punctuation_density_all": total / words * 100,
        "punctuation_density_nonterminal": (
            nonterminal / words * 100
        ),
    }


# ---------------------------------------------------------
# FEATURE 2: Punctuation per sentence
# ---------------------------------------------------------

def punctuation_per_sentence(text):
    """
    Return average punctuation marks per sentence.

    Returns both:
      - punctuation_per_sentence_all
      - punctuation_per_sentence_nonterminal
    """
    sentences = count_sentences(text)

    if sentences == 0:
        return {
            "punctuation_per_sentence_all": float("nan"),
            "punctuation_per_sentence_nonterminal": float("nan"),
        }

    total = count_all_punctuation(text)
    nonterminal = count_nonterminal_punctuation(text)

    return {
        "punctuation_per_sentence_all": total / sentences,
        "punctuation_per_sentence_nonterminal": (
            nonterminal / sentences
        ),
    }


# ---------------------------------------------------------
# FEATURE 3: Punctuation-specific rates per 100 words
# ---------------------------------------------------------

def punctuation_specific_rates(text):
    """
    Return the rate of each selected punctuation category
    per 100 words.
    """
    words = count_words(text)
    counts = get_punctuation_counts(text)

    if words == 0:
        return {
            f"rate_{name}": float("nan")
            for name in counts
        }

    return {
        f"rate_{name}": count / words * 100
        for name, count in counts.items()
    }


# ---------------------------------------------------------
# FEATURE 4: Relative punctuation composition
# ---------------------------------------------------------

def relative_punctuation_composition(text):
    """
    Return the proportion of each tracked punctuation type
    among all punctuation marks in the tracked categories.

    These proportions sum to 1 when at least one tracked
    punctuation mark is present.
    """
    counts = get_punctuation_counts(text)
    total_tracked = sum(counts.values())

    if total_tracked == 0:
        return {
            f"composition_{name}": float("nan")
            for name in counts
        }

    return {
        f"composition_{name}": count / total_tracked
        for name, count in counts.items()
    }


# ---------------------------------------------------------
# Main driver
# ---------------------------------------------------------

def run_analysis():
    import os

    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"Text", "Bias", "StoryID"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Normalize labels while preserving the original data.
    df["Bias"] = df["Bias"].astype("string").str.strip().str.lower()

    bias_mapping = {
        "left": "Left",
        "left-leaning": "Left",
        "center": "Center",
        "centre": "Center",
        "center-leaning": "Center",
        "right": "Right",
        "right-leaning": "Right",
    }

    df["Bias"] = df["Bias"].map(bias_mapping).fillna(df["Bias"])

    # Treat missing text as empty for feature extraction.
    df["Text"] = df["Text"].fillna("").astype(str)

    # Calculate the basic counts once per record.
    df["word_count"] = df["Text"].apply(count_words)
    df["sentence_count"] = df["Text"].apply(count_sentences)
    df["punctuation_count"] = df["Text"].apply(
        count_all_punctuation
    )

    
    # Calculate title features separately.
    df["Title"] = df["Title"].fillna("").astype(str)

    df["title_word_count"] = df["Title"].apply(count_words)
    df["title_punctuation_count"] = df["Title"].apply(
        count_all_punctuation
    )

    # Title punctuation density
    title_density = df["Title"].apply(punctuation_density)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {f"title_{k}": v for k, v in features.items()}
                    for features in title_density
                ],
                index=df.index,
            ),
        ],
        axis=1,
    )

    # Title-specific punctuation rates
    title_rates = df["Title"].apply(punctuation_specific_rates)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {f"title_{k}": v for k, v in features.items()}
                    for features in title_rates
                ],
                index=df.index,
            ),
        ],
        axis=1,
    )

    # Title relative punctuation composition
    title_composition = df["Title"].apply(
        relative_punctuation_composition
    )
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {f"title_{k}": v for k, v in features.items()}
                    for features in title_composition
                ],
                index=df.index,
            ),
        ],
        axis=1,
    )

    # Apply the four feature functions.
    feature_groups = [
        df["Text"].apply(punctuation_density),
        df["Text"].apply(punctuation_per_sentence),
        df["Text"].apply(punctuation_specific_rates),
        df["Text"].apply(relative_punctuation_composition),
    ]

    for feature_group in feature_groups:
        features = pd.DataFrame(
            feature_group.tolist(),
            index=df.index
        )
        df = pd.concat([df, features], axis=1)

    # Save record-level results.
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nDataset shape:", df.shape)
    print("\nRecords per category:")
    print(df["Bias"].value_counts(dropna=False))

    # Summary statistics for the main features.
    summary_features = [
        "punctuation_density_all",
        "punctuation_density_nonterminal",
        "punctuation_per_sentence_all",
        "punctuation_per_sentence_nonterminal",
    ]

    print("\nSummary statistics:")
    print(
        df.groupby("Bias")[summary_features]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )

    # Check repeated article identifiers.
    print("\nStoryID audit:")
    print("Total records:", len(df))
    print("Unique StoryIDs:", df["StoryID"].nunique())
    print("Records with repeated StoryIDs:",
          df["StoryID"].duplicated(keep=False).sum())

    # -------------------------
    # Plot 1: Punctuation density
    # -------------------------
    plot_boxplots(
        df,
        [
            "punctuation_density_all",
            "punctuation_density_nonterminal",
        ],
        "Punctuation Density by Political Leaning",
        "Punctuation marks per 100 words",
        "01_punctuation_density.png",
    )

    # -------------------------
    # Plot 2: Punctuation per sentence
    # -------------------------
    plot_boxplots(
        df,
        [
            "punctuation_per_sentence_all",
            "punctuation_per_sentence_nonterminal",
        ],
        "Punctuation per Sentence by Political Leaning",
        "Punctuation marks per sentence",
        "02_punctuation_per_sentence.png",
    )

    # -------------------------
    # Plot 3: Individual punctuation rates
    # -------------------------
    rate_columns = [
        f"rate_{name}" for name in PUNCTUATION_PATTERNS
    ]

    plot_feature_panels(
        df,
        rate_columns,
        "Punctuation-Specific Rates by Political Leaning",
        "Occurrences per 100 words",
        "03_punctuation_specific_rates.png",
    )

    # -------------------------
    # Plot 4: Relative composition
    # -------------------------
    composition_columns = [
        f"composition_{name}"
        for name in PUNCTUATION_PATTERNS
    ]

    plot_feature_panels(
        df,
        composition_columns,
        "Relative Punctuation Composition by Political Leaning",
        "Proportion of tracked punctuation",
        "04_relative_punctuation_composition.png",
    )

    print(f"\nResults saved to: {OUTPUT_CSV}")
    print(f"Plots saved in: {PLOT_DIR}/")


# ---------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------

def plot_boxplots(df, columns, title, ylabel, filename):
    plot_df = df.melt(
        id_vars=["Bias"],
        value_vars=columns,
        var_name="Feature",
        value_name="Value",
    )

    plot_df = plot_df.dropna(subset=["Value"])

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=plot_df,
        x="Feature",
        y="Value",
        hue="Bias",
        order=columns,
        showfliers=False,
    )

    plt.title(title)
    plt.xlabel("")
    plt.ylabel(ylabel)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{filename}", dpi=200)
    plt.close()


def plot_feature_panels(df, columns, title, ylabel, filename):
    """
    Plot each feature in a separate panel so features with
    different scales remain interpretable.
    """
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 4.5 * nrows),
    )

    axes = axes.flatten()

    for ax, column in zip(axes, columns):
        sns.boxplot(
            data=df,
            x="Bias",
            y=column,
            order=["Left", "Center", "Right"],
            showfliers=False,
            ax=ax,
        )
        ax.set_title(column.replace("_", " ").replace("rate ", ""))
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)

    for ax in axes[len(columns):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=16, y=1.002)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/{filename}", dpi=200,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_analysis()
