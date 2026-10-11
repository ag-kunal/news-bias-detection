# Sentiment feature experiment

This module runs a standalone sentiment-only baseline on each dataset CSV separately, including `Combined_Dataset.csv`.

## Features

- `vader_compound_mean`: mean VADER compound score over sentences in an article.
- `vader_positive_mean`, `vader_negative_mean`, `vader_neutral_mean`: mean sentence-level VADER proportions.
- `positive_polar_word_count`, `negative_polar_word_count`: occurrence counts for text tokens in VADER's lexicon with scores >= +2 or <= -2 respectively.
- `highly_polar_word_count`: sum of positive and negative polar-word counts.
- Three `*_per_1000_words` features: the corresponding polar-word count per 1,000 alphabetic word tokens.

The +2/-2 thresholds are explicit project choices for strongly polar words, not VADER's compound-score sentiment-label thresholds. Sentiment is not the same thing as political bias; the classifier measures how predictive these sentiment-derived features are of the dataset's existing `Bias` labels.

## Run

From the repository root, create and activate a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r sentiment/requirements.txt
```

Extract features for all CSVs in `data/IRE_Dataset/`:

```bash
python sentiment/extract_features.py
```

Train/evaluate a separate Logistic Regression baseline for each enriched dataset:

```bash
python sentiment/train_baseline.py
```

## Outputs (local only)

- `outputs/sentiment/features/*_sentiment.csv`: original data columns plus `Article_ID` and sentiment features.
- `outputs/sentiment/features/feature_extraction_summary.csv`: input/output row counts.
- `outputs/sentiment/results/sentiment_results.csv`: one summary row per dataset.
- `outputs/sentiment/results/sentiment_classification_report.csv`: precision, recall, F1, and support per class.
- `outputs/sentiment/results/confusion_matrices/`: confusion matrices by dataset.

The source data is never modified. IDs are generated from the source filename and row order, so keep the source CSV order fixed to preserve them.

## Evaluation notes

The classifier uses only the sentiment features (never `Title`, `Text`, `Source`, or any other non-sentiment columns as predictors). `Bias` is the target. Each dataset is evaluated separately using the same 80/20 stratified train/test split seed, a standard-scaled Logistic Regression model, and balanced class weights. Macro-F1 and per-class metrics should be considered alongside accuracy, particularly due to class imbalance. The combined dataset may overlap with the individual datasets, so treat its score as a separate dataset experiment rather than independent confirmation.
