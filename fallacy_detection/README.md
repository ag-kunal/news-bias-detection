# Stage 2: fallacy-detection features (feasibility prototype)

Section 3.1.6 of the draft proposes fallacy features for media-bias detection. This module tests whether that is
feasible with the corpora from the papers the professor shared.

## What changed from the first prototype

The first prototype scored sentences against 12 free-text fallacy descriptions with a zero-shot NLI model
(`facebook/bart-large-mnli`). MAFALDA and FAINA both report that zero-shot models do very poorly on fallacy
detection (MAFALDA Level-2 F1 about 0.1 for GPT-3.5; FAINA span-level F1 of 3 to 6), and the NLI approach is also slow:
12 labels x sentences x articles forward passes on CPU. It is kept only as an optional comparison
(`extract_features_zeroshot.py`). The default is now a supervised model trained on the labelled corpora.

## Pipeline

```bash
# 0. one-time: clone the labelled corpora into fallacy_detection/reference_data/
mkdir -p fallacy_detection/reference_data && cd fallacy_detection/reference_data
git clone --depth 1 https://github.com/ChadiHelwe/MAFALDA
git clone --depth 1 https://github.com/pierpaologoffredo/FallacyDetection
git clone --depth 1 https://github.com/causalNLP/logical-fallacy
cd ../..

python fallacy_detection/prepare_fallacy_data.py        # unified sentence-level training table
python fallacy_detection/train_fallacy_classifier.py    # detector + type model, writes metrics
python fallacy_detection/extract_features.py --limit 50 # smoke test on the news data, then drop --limit
python fallacy_detection/make_annotation_sample.py --input data/Combined_Dataset.csv   # blind sample for manual labels
```

Output: `outputs/fallacy_detection/<dataset>_fallacy_features.csv` (Article_ID, Bias, features only; no article text).

## Sources and how labels were harmonised

| Corpus | Paper | Used for | Notes |
|---|---|---|---|
| MAFALDA gold standard (200 texts, sentence labels) | Helwe et al., NAACL 2024 | detector + typer | has real "no fallacy" sentences. Alternatives: a sentence counts as fallacy only if some label set lacks `nothing` |
| ElecDeb60to20 (US debates) | Goffredo et al., EMNLP 2023 | detector + typer | negatives are neighbouring context sentences that do not overlap any annotated fallacy; Slogans dropped |
| LOGIC edu + climate | Jin et al., 2022 | typer only | positives only, so it cannot teach "no fallacy"; umbrella labels (relevance, logic, credibility, intentional) dropped from typing |

FAINA (Ramponi et al., NAACL 2025) is Italian and was not used for training, but motivated two choices: keep ambiguity visible
(MAFALDA alternatives are not forced to one label) and report transfer, not just in-corpus scores.

Working types: ad hominem, appeal to authority, appeal to emotion, causal oversimplification (false cause), slippery slope,
hasty generalization, false dilemma, ad populum, circular reasoning, other. The draft's three features
(hasty generalization, appeal to emotion, ad hominem) are among them.

## How to read the results (run it and check `outputs/fallacy_detection/training/`)

Run on the cloned corpora, the detector scores about F1 0.63 / AUC 0.75 in pooled grouped CV, but only AUC about 0.56 when trained on one
corpus and tested on the other (leave-one-source-out). The type model drops from macro-F1 0.52 (pooled) to 0.2 to 0.35 when
tested on a corpus it never saw. In other words, **the signal learned from debates and Reddit does not transfer reliably, and
news articles are a further domain shift.** This matches the weak numbers in the papers (Cantin and Chust report binary F1 0.34 on
MM-USED-fallacy). Treat the article features as weak, noisy proxies.

On synthetic placeholder text the detector flagged over 80% of sentences, so on real news check
`overall_flagged_rate` in `run_manifest.csv`; if it is that high, prefer the soft features (`*_soft_share`, `fallacy_p_mean`)
over thresholded ones, or raise the threshold.

## Before using any number in the paper

1. Run `make_annotation_sample.py`, have two annotators label the blind sample, report Cohen's kappa.
2. Compare model flags against the human labels (precision/recall per type).
3. Compare Stage 1 vs Stage 1 + fallacy with `feature_pipeline/evaluate_baseline.py --group-by source`.
4. Natural next step if the above shows signal: fine-tune a transformer (RoBERTa or BERT, as in Cantin and Chust, Goffredo) on the same
   `fallacy_sentences.csv`. It was not done here because it needs a GPU and model downloads.
