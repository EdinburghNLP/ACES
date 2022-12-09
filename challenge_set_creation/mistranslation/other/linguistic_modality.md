###Linguistic Modality:

**Construction:** Semi-automatic

**Aim:** Construct examples to assess whether metrics can identify when modal auxiliary verbs are incorrectly translated.

**Note:** We focus on German to English translation for the English modal auxiliary verbs: "must" (necessity), and "may",
"might", "could" (possibility).

**Instructions:**

1. Run the python script perturb-modals.py to identify parallel sentences where there is a modal verb in the English
reference sentence (from our list of English modals above) and to translate the (German) source sentence using
Google Translate to obtain a translation (in English). The script will output two lines for each input sentence pair:
a deletion example and a substitution example.
`python perturb-modals.py --in-file test/modals_test.tsv --out-file test/modals_test_out.tsv`
2. Manually verify examples tagged as "deletion", i.e. where the modal is deleted.
3. Manually construct examples where the modal is substituted, by modifying the MT output in the incorrect translation.
	a. Manually verify the "good" translation.
	b. Manually alter the "incorrect" translation to substitute the (English) modal verb with one that conveys a
	different meaning or epistemic strength e.g. replace "might" (possibility) with "will", which denotes (near)
	certainty.
4. Exclude any instances of "may" with deontic meaning (e.g. expressing permission) from the set, leaving only those
instances of "may" with an epistemic meaning (expressing probability or prediction).

**Example:**

>**SRC (de):** Mit der Einführung dieser Regelung könnte diese Freiheit enden.

>**REF (en):** With this arrangement in place, this freedom might end.

>**good:** With the introduction of this regulation, this freedom could end.

>**incorrect:** With the introduction of this regulation, this freedom will end.