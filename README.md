# build-it-break-it-22


## Datasets
As the created files are greater than the allowed repository capacity, find the datasets [here](https://uoe-my.sharepoint.com/:f:/r/personal/s1948463_ed_ac_uk/Documents/Work/Build-It-Break-It-22?csf=1&web=1&e=gglbyX)

### XNLI

...


## Command Line Interface

### Installation

For editable installation:

    pip install -e .

Depending on which languages you process, you will need to install the corresponding spacy models.

### Creating Adversarial Examples

Input: One or more TSV files with 'source', 'good-translation' and 'reference' fields.

You can automatically create adversarial examples by running:

    breakit-perturb -i your_file.tsv ... -l target_lang

Output: One TSV file per input file named 'your_file.perturbed.tsv' with 'source', 'good-translation', 'incorrect-translation', 'reference', 'phenomena' fields.

**Currently implemented perturbations:**

- Negation deletion and double negation
- Unit conversions (only for English)
- Number changes
- Named entity changes

**Currently supported languages:**

- English
- German
- French
- Spanish
- Korean
- Japanese
- Chinese

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
