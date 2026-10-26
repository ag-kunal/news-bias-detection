# Bias-inducing language / title-content congruence

The available paper text under Section 3.1.3 describes measuring cosine similarity between a title embedding and article sentence embeddings using Sentence Transformers, then averaging similarities as a topic-similarity measure. It does not specify a model checkpoint or an operational definition of “keyphrase.” This module implements the stated title-content similarity method with an explicit, replaceable checkpoint; it does **not** invent a supposedly authoritative list of bias-inducing words.

Install the optional dependency and run from the repository root:

```bash
python -m pip install sentence-transformers
python bias_inducing_words/extract_features.py --model sentence-transformers/all-MiniLM-L6-v2
```

Input CSVs must contain title/headline and text/content columns. Output is written to `outputs/bias_inducing_words/`. The checkpoint is recorded in the run manifest. `title_content_similarity_low_0_3` uses a 0.3 threshold only as an exploratory convenience; the paper does not define this threshold, so do not treat it as a paper-derived threshold or as a definitive bias label.

The method computes cosine similarity of the title embedding with each sentence embedding and records mean, median, min, and max. This is a reproducible operationalization of the text in the draft, not proof that the title is misleading or the article is biased. Confirm the exact model and whether keyphrase extraction is intended with the professor before making claims of exact replication.
