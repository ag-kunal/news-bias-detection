# Punctuation Feature Experiment

Extracts article-body punctuation features from each CSV in `data/`, preserves all source columns and rows, and compares feature distributions across the existing `Bias` labels.

## Features

- Counts for periods, commas, question marks, exclamation marks, semicolons, colons, single/double quotation marks, parentheses, square brackets, dashes/hyphens, and ellipses.
- Per-1,000-word rates for each punctuation type, total punctuation, and combined question/exclamation marks.
- Per-label count, mean, median, and standard deviation summaries.
- Kruskal–Wallis tests across available bias labels, with Benjamini–Hochberg correction across the tested features within each dataset.

The script analyzes `Text` (article body) rather than `Title`. Normalization uses the script's documented word-token regex. These are descriptive/style features; a statistically significant difference does not establish that punctuation causes or proves political bias.

## Run

Install dependencies:

```bash
python -m pip install pandas numpy matplotlib scipy
```

From the repository root:

```bash
python Punctuation/punctuation_analysis.py
```

Optional paths:

```bash
python Punctuation/punctuation_analysis.py --input-dir data --output-dir outputs/punctuation
```

The input directory is scanned for CSV files, so `Combined_Dataset.csv` is processed when present. Original datasets are never modified.

## Outputs

- `outputs/punctuation/features/*_punctuation.csv`: original columns plus `Article_ID` and extracted features.
- `outputs/punctuation/summaries/*_by_bias.csv`: descriptive statistics by bias label.
- `outputs/punctuation/summaries/*_significance_tests.csv`: per-feature Kruskal–Wallis tests and adjusted p-values.
- `outputs/punctuation/plots/*_punctuation_rates.png`: plot for each dataset.
- `outputs/punctuation/punctuation_run_manifest.csv`: row counts and output paths.
- `outputs/punctuation/punctuation_significance_tests_all_datasets.csv`: combined significance-test report.
