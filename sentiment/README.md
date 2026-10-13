# Sentiment Feature Experiment

This module extracts VADER sentiment features and evaluates a sentiment-only classifier independently on each of the five CSV datasets, including `Combined_Dataset.csv`.

## Features

| Feature | Meaning |
|---|---|
| `vader_compound_mean` | Mean VADER compound score across an article's sentences, ranging from -1 to +1. |
| `vader_positive_mean` | Mean sentence-level positive proportion from VADER. |
| `vader_negative_mean` | Mean sentence-level negative proportion from VADER. |
| `vader_neutral_mean` | Mean sentence-level neutral proportion from VADER. |
| `positive_polar_word_count` | Occurrences of tokens whose VADER lexicon score is at least +2. |
| `negative_polar_word_count` | Occurrences of tokens whose VADER lexicon score is at most -2. |
| `highly_polar_word_count` | Sum of positive and negative polar-word counts. |
| `positive_polar_per_1000_words` | Positive polar-word count per 1,000 alphabetic word tokens. |
| `negative_polar_per_1000_words` | Negative polar-word count per 1,000 alphabetic word tokens. |
| `highly_polar_per_1000_words` | Total polar-word count per 1,000 alphabetic word tokens. |

The +2/-2 lexicon thresholds are project choices for identifying strongly polar words; they are not VADER's compound-score class thresholds. Sentiment is not the same as political bias. The classifier tests whether these features help predict the datasets' existing `Bias` labels.

## Setup

Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r sentiment/requirements.txt
```

Place the five approved source CSVs in `data/IRE_Dataset/`.

## Run the Pipeline

### 1. Extract features

```bash
python sentiment/extract_features.py --output-dir sentiment/outputs/features
```

Creates an enriched CSV per input dataset and `feature_extraction_summary.csv`. The source CSVs are not modified, and all input rows are retained by the extractor.

### 2. Train sentiment-only baselines

```bash
python sentiment/train_baseline.py \
  --features-dir sentiment/outputs/features \
  --results-dir sentiment/outputs/results
```

Each dataset is evaluated separately with a standard-scaled Logistic Regression classifier, balanced class weights, a stratified 80/20 holdout split, and fixed random seed. Only the sentiment columns are predictors; `Bias` is the target.

Outputs include:

- `sentiment/outputs/results/sentiment_results.csv`: one summary row per dataset, including macro-F1, accuracy, balanced accuracy, and macro precision/recall.
- `sentiment/outputs/results/sentiment_classification_report.csv`: per-class precision, recall, F1, and support.
- `sentiment/outputs/results/confusion_matrices/`: confusion matrices as CSV files.

### 3. Generate box plots

```bash
python sentiment/plot_features.py \
  --features-dir sentiment/outputs/features \
  --output-dir sentiment/outputs/results/boxplots
```

The plotting script creates four box plots per dataset, grouped by `Bias`: mean VADER compound score, positive polar words per 1,000 words, negative polar words per 1,000 words, and total polar words per 1,000 words. These plots are exploratory summaries; they do not demonstrate causation or prove that a feature improves classification.

## Data and Git Policy

- Keep original CSVs in `data/IRE_Dataset/` locally; do not commit them.
- Keep enriched feature CSVs and extraction summaries in `sentiment/outputs/features/` locally; do not commit them.
- Commit code and documentation. Commit only the final, useful model summary tables and plots under `sentiment/outputs/results/` when appropriate to share.
- Check `git status` before every commit to ensure no original or enriched data files are staged.

## Interpretation Notes

Macro-F1 and per-class metrics matter because the label distribution is imbalanced. Compare dataset scores cautiously because datasets differ in size and content. `Combined_Dataset.csv` may overlap with the individual datasets, so its score should be treated as a separate experiment rather than independent confirmation.
