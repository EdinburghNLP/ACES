###Punctuation:

**Construction:** Automatic

**Aim:** Perturb punctuation for translated sentences (manually or automatically translated). There are four strategies for
this, which will be applied:

punctuation:deletion_all : Strip all punctuation
punctuation:deletion_quotes : Strip all quotes
punctuation:deletion_commas : Strip all commas
punctuation:statement-to-question : Turn statements ending with exclamation marks, into questions (!->?).

**Notes:**

* The translated sentence will be the "good" translation, and its perturbed version will be the "incorrect"
translation. The perturbations are designed to produce a "worse" translation.
* Order in which rules are applied: statement-to-question, deletion_quotes, random: deletion_commas | deletion_all.

**Instructions:**

1. Construct a tab-separated file (with header) containing three columns: source, reference, "good" translation.
2. Run script: perturb_punctuation.py using the following command:
`python perturb_punctuation.py --in-file test/punctuation_test.tsv --out-file test/punctuation_test_out.tsv`