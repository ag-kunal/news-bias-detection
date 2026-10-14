# Adjective Title Feature Experiment

Extracts adjective counts from news titles using spaCy's English part-of-speech tags, then summarizes and compares the features by the datasets' existing `Bias` labels. Adjective counts are linguistic features, not evidence that a title or publisher is politically biased.

## Feature Definitions

| Feature | Definition |
|---|---|
| `title_adjective_count` | Number of alphabetic title tokens tagged `ADJ` by the selected spaCy model. Occurrences are counted, not unique words. |
| `title_pos_token_count` | Number of alphabetic tokens from the same spaCy document; this is the normalization denominator. |
| `title_adjective_rate_per_100_tokens` | `100 * title_adjective_count / title_pos_token_count`; undefined when the denominator is zero. |
| `title_has_adjective` | Whether at least one token is tagged `ADJ`. |
| `title_is_valid` | Whether the source title is nonblank. |

The default tagger is `en_core_web_sm`. Record the model and spaCy versions in `extraction_summary.csv`; use the same model version for comparable reruns. No titles are manually filtered, and source datasets are never modified.

## Setup

Run from the repository root:

```bash
python -m pip install -r Adjective/requirements.txt
python -m spacy download en_core_web_sm
```

The five approved CSVs should be in `data/`. By default, extraction processes all CSV files in that directory.

## Run

Extract title features:

```bash
python Adjective/extract_features.py
```

For a smoke test on one dataset:

```bash
python Adjective/extract_features.py \
  --input-file data/Combined_Dataset.csv \
  --limit 100
```

Run grouped summaries, comparisons, and plots:

```bash
python Adjective/analyze_and_plot.py
```

Extraction writes one row-preserving `*_adjectives.csv` per input plus `extraction_summary.csv` under `Adjective/outputs/features/`. Analysis writes `adjective_bias_summary.csv`, `adjective_group_comparisons.csv`, and one rate plot per dataset under `Adjective/outputs/results/`. Pass `--features-dir` or `--results-dir` to override those paths.

## Analysis Notes

- Summaries are calculated separately for each dataset and existing bias label. They include title counts, rate distribution statistics, the percentage of titles containing an adjective, and cluster-bootstrap 95% intervals for the mean rate.
- Datasets without repeated `StoryID`s receive Kruskal-Wallis and pairwise Mann-Whitney tests; pairwise p-values are Holm-adjusted.
- `Combined_Dataset.csv` has repeated `StoryID`s. Its comparisons use StoryID-cluster bootstrap confidence intervals for differences in mean adjective rate and do not report row-level p-values. Repeated stories remain in the derived data and are not deduplicated.
- Dataset results should not be pooled as independent evidence: `Combined_Dataset.csv` can overlap the individual datasets. The combined file has multiple sources, so source-adjusted modeling is a separate follow-up analysis.

Run the focused feature tests with:

```bash
python -m unittest discover -s Adjective -p 'test_*.py'
```