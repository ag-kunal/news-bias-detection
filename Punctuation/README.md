
# Punctuation Feature Experiment

This module analyzes punctuation usage across political news datasets to identify differences between Left, Center, and Right bias categories using statistical summaries and box plots.

## Features

- **Punctuation Density:** Punctuation marks per 100 words.

  - Formula: `Total punctuation / Total words * 100`
  - Includes all punctuation and a variant excluding `.`, `?`, and `!`.
- **Punctuation per Sentence:** Average punctuation marks per sentence.

  - Formula: `Total punctuation / Total sentences`
  - Includes all punctuation and a variant excluding `.`, `?`, and `!`.
- **Punctuation-Specific Rates:** Occurrences of individual punctuation types per 100 words, including commas, semicolons, colons, dashes, parentheses, quotation marks, question marks, exclamation marks, and periods.
- **Relative Punctuation Composition:** Proportion of each punctuation type among all tracked punctuation marks.

  - Formula: `Count of punctuation type / Total tracked punctuation`

Title and body punctuation can be analyzed separately. Punctuation per sentence is primarily intended for body text.

## Run

Install dependencies:

```bash
pip install pandas nltk matplotlib seaborn
```

Run the analysis:

```bash
python punctuation/punctuation_analysis.py
```

Update the script and dataset paths if necessary.

## Outputs

- `punctuation_analysis_results.csv`: Original data and extracted punctuation features.
- `01_punctuation_density.png`: Punctuation density box plots.
- `02_punctuation_per_sentence.png`: Punctuation per sentence box plots.
- `03_punctuation_specific_rates.png`: Individual punctuation-rate comparisons.
- `04_relative_punctuation_composition.png`: Punctuation composition comparisons.

## Analysis

The extracted features are compared across the existing `Bias` labels using descriptive statistics and box plots. Statistical tests can be used to assess whether the observed differences are significant.

Punctuation patterns are stylistic indicators and do not independently establish political bias.
