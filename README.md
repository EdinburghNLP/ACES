# ACES: Translation Accuracy Challenge Sets for Evaluating Machine Translation Metrics

As machine translation (MT) metrics improve their correlation with human judgement every year, it is crucial to understand the limitations of these metrics at the segment level. Specifically, it is important to investigate metric behaviour when facing accuracy errors in MT because these can have dangerous consequences in certain contexts (e.g., legal, medical). We curate ACES, a translation accuracy challenge set, consisting of 68 phenomena ranging from simple perturbations at the word/character level to more complex errors based on discourse and real-world knowledge. 
We use ACES to evaluate a wide range of MT metrics including the submissions to the WMT 2022 metrics shared task and perform several analyses leading to general recommendations for metric developers. We recommend: a) combining metrics with different strengths, b) developing metrics that give more weight to the source and less to surface-level overlap with the reference and c) explicitly modelling additional language-specific information beyond what is available via multilingual embeddings.

## Command Line Interface

### Installation

For editable installation:

    pip install -e .

### Scoring Adversarial Examples

Input: One or more TSV files with 'source', 'good-translation', 'incorrect-translation', 'reference', 'phenomena' fields.

You can automatically score adversarial examples by running:

    breakit-score -i your_file.tsv ...

Output: One TSV file per input file named 'your_file.scored.tsv' with 'source', 'good-translation', 'incorrect-translation', 'reference', 'phenomena' fields and a 'metric-good' and 'metric-bad' field for every metric that stores the sentence-level scores.

**Currently supported metrics:**

- BLEU
- ChrF
- COMET (needs GPU access)
- COMET-QE (needs GPU access)

### Evaluating Metrics

Input: TSV file with 'source', 'good-translation', 'incorrect-translation', 'reference', 'phenomena' fields and metric scores in 'metric-good' and 'metric-bad' fields.

You can evaluate metrics by running:

    breakit-eval -i your_file.tsv ...

Output: STDOUT overview of Kendall Tau scores per file, phenomenon and metric.

## Coming Soon:

- code for automatic perturbations we implemented ourselves
- explanations for automatic perturbations that use other libraries
- guidelines for manual creation/annotation of examples

## Citation

@inproceedings{amrhein-aces-2022,
title = "{ACES}: Translation Accuracy Challenge Sets for Evaluating Machine Translation Metrics",
author = {Amrhein, Chantal  and
  Moghe, Nikita and
  Guillou, Liane},
booktitle = "Seventh Conference on Machine Translation (WMT22)",
month = dec,
year = "2022",
address = "Abu Dhabi, United Arab Emirates",
publisher = "Association for Computational Linguistics",
eprint = {2210.15615}
}
