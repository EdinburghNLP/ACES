###Pronouns:

**Construction:** Semi-automatic

**Aim:** Construct examples where the pronoun in the "incorrect" translation is either deleted or replaced with an
inappropriate translation.

**Notes:**

* The instructions assume that you already have a PROTEST-style QSL database of MT translations and annotations over the
pronoun translations. If you need to construct this database from scratch, please follow the instructions at the end.
* The python script extract-pronoun-translation-examples.py is designed to perturb pronouns in German MT output. If your
target language is not German you will need to define your own bad pronoun translation mappings in the script:
bad_pro_trans_map.

**Instructions:**

1. Open your PROTEST database and run the SQL SELECT statement in extract_protest_annotations.sql. Save the file as
pronoun_examples.csv.
2. Run the python script extract-pronoun-translation-examples.py using the following command:
`python extract-pronoun-translation-examples.py --in-file test/pronoun_examples.csv --outfile test/pronoun_examples_out.tsv`
3. Note that the sentences extracted from a PROTEST-style database will be tokenised. For ACES-style examples, please
use your favourite detokeniser script/method to detokenise the source/reference/MT sentences.


To construct an use a PROTEST-style test-suite from scratch:

1. Follow the guidelines from ParCor [0] or ParCorFull [1], to manually annotate the pronouns in a parallel corpus.
2. Construct a PROTEST-style [2] test suite of example pronouns.
3. Run MT systems to translate the test suit examples.
4. Collect manual judgements for each of the translated test suite examples.

**[0]** Liane Guillou, Christian Hardmeier, Aaron Smith, Jörg Tiedemann, and Bonnie Webber. (2014). ParCor 1.0: A Parallel
	Pronoun-Coreference Corpus to Support Statistical MT. In proceedings of the Ninth International Conference on
	Language Resources and Evaluation (LREC'14), pages 3191–3198, Reykjavik, Iceland.

**[1]** Ekaterina Lapshinova-Koltunski, Christian Hardmeier, and Pauline Krielke. (2018). ParCorFull: a Parallel Corpus
	Annotated with Full Coreference. In proceedings of the Eleventh International Conference on Language Resources and
	Evaluation (LREC 2018), pages 423-428, Miyazaki, Japan.

**[2]** Liane Guillou and Christian Hardmeier. (2016). PROTEST: A Test Suite for Evaluating Pronouns in Machine Translation
	In proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC'16), pages 636–643,
	Portorož, Slovenia.