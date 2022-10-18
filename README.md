# build-it-break-it-22




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
