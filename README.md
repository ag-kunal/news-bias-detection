# News Bias Detection

Research project exploring whether linguistic and sentiment-based features can help classify news articles by bias label.

## Dataset

The datasets are stored separately from this repository. See [`data/README.md`](data/README.md) for access details and the expected dataset format.

## Planned Feature Extraction

- Sentiment features (VADER)
- Adjective features ([`Adjective/`](Adjective/README.md))
- Punctuation features
- Readability and lexical features ([`readability/`](readability/README.md))
- Feature combination and ML evaluation

Feature-extraction scripts and experiments will be added as the project progresses.

## Collaboration Guidelines

- Use the agreed cleaned dataset version shared by the team.
- Do not independently remove rows or change labels.
- Preserve article records and use a stable article ID when combining feature outputs.
- Document feature definitions, libraries, formulas, and thresholds.
- Evaluate models with macro-F1 and per-class precision/recall, in addition to accuracy.
