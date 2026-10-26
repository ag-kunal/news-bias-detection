# Feature pipeline

Run from the repository root after running the individual extractors:

```bash
python feature_pipeline/merge_features.py
python feature_pipeline/evaluate_baseline.py
```

`merge_features.py` merges derived feature columns on `Article_ID`, checks one-to-one joins, preserves source row counts, and writes a manifest. It will not join a feature file that lacks IDs. IDs are based on dataset filename plus source row order; if a stable source article ID exists, update the extractors to use it consistently.

`evaluate_baseline.py` evaluates numerical features with a stratified 80/20 split and Logistic Regression. It writes `baseline_metrics.csv` and `classification_reports.json`. Source is excluded by default because it can leak publisher identity into bias-label prediction. This is a preliminary diagnostic, not a final benchmark. Deduplicate cross-dataset overlaps and use group-based splits before reporting paper results.
