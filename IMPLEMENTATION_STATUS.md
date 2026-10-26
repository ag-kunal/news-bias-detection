# Stage 1 implementation status

## Implemented in this repository snapshot

- Punctuation counts/rates, descriptive summaries, plots, Kruskal–Wallis tests and Benjamini–Hochberg correction.
- Existing VADER sentiment extraction, sentiment-only baseline, and feature-association analysis.
- Lexical/readability extraction: counts, lexical diversity, stopwords, numeric tokens, approximate readability metrics, hapax and Yule's K.
- Optional spaCy parser features with explicit naming of proxy features.
- Section 3.1.3 title/content topic-congruence features using Sentence Transformers: title-to-sentence cosine similarity and article-level summary statistics.
- Consistent README and root dependency manifest. Source CSVs are read-only inputs; derived outputs are separate.

## Validation completed here

- Python syntax compilation passed for all Python scripts.
- Full lexical extraction over the four supplied datasets was attempted, but exceeded this environment's execution time. Do not interpret that timeout as successful full-dataset validation.
- The title/content embedding pipeline was not run end-to-end here; `sentence-transformers` and its model may need to be installed/downloaded.
- Earlier punctuation runs in this workspace reported row-count preservation for the four component datasets. The combined dataset was not supplied.

## Remaining before claiming exact reproduction of the paper

1. **Sentiment parity:** the draft describes LIWC subjectivity, Bing/Subjectivity polarity lexicons, and selected high-intensity VADER words. Current implementation is VADER-based and is not equivalent. Required licensed/approved lexicons, tokenization, and thresholds need confirmation.
2. **Syntactic parity:** the draft describes dependency-tree length and phrase/clause tags. Optional spaCy features here are approximate and not exact replicas of the specified parser/tagset methodology.
3. **Section 3.1.3 details:** the draft describes mean title-to-sentence embedding cosine similarity, but does not name the model checkpoint or specify the keyphrase method. Current default `all-MiniLM-L6-v2` is a documented implementation choice, not a paper-specified fact. Confirm with the professor.
4. **Full data validation:** run every script on the approved datasets in the user's local environment, confirm input/output row counts and inspect logs. `Combined_Dataset.csv` was not supplied in this workspace.
5. **Evaluation integrity:** ensure duplicate/overlapping articles do not leak across train/test splits and report macro-F1 plus per-class precision/recall.

## Suggested local verification

```bash
python -m pip install -r requirements.txt
python Punctuation/punctuation_analysis.py
python sentiment/extract_features.py
python sentiment/analyze_feature_associations.py
python lexical_readability/extract_features.py
python bias_inducing_words/extract_features.py --model sentence-transformers/all-MiniLM-L6-v2
```

If you have a validated source identifier, use it when merging feature tables instead of relying on generated row-order IDs.
