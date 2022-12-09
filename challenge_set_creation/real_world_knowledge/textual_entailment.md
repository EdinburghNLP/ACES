###Textual entailment [VERBS]

**Construction:** Manual

**Aim:** Construct examples where the meaning of the source/reference is entailed by the "good" translation
Note: we are interested in directional entailment, therefore paraphrases, which are directional, should be excluded.

**Instructions:**

1. Identify a list of predicate (i.e. verb) pairs for which an entailment relation is possible (e.g. win -> play,
buy -> own, was murdered -> died).
2. For a given predicate pair P (p -> q), obtain a source sentence S and a reference translation R, where R contains a
mention of predicate "p" (e.g. "was murdered" for the entailment: was murdered -> died).
3. Construct the "good" translation: copy R and replace the instance of predicate "p" with the entailed predicate "q"
(e.g. "died").
4. Construct the "incorrect" translation: copy R and replace the instance of predicate "p" with a related but
non-entailed predicate "x" (e.g. "was attacked").

**Example:**

>**SRC (de):** Ein Mann wurde ermordet.

>**REF (en):** A man was murdered.

>**good:** A man died.

>**incorrect:** A man was attacked.

In cases where an antonym of the verb is available, use the antonym in the incorrect translation.  e.g. if "lost" is in
the source/reference, use "won" in the incorrect translation (lost -/> won).