#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from csv import reader

random.seed(42)


# Mapping of German pronouns to options for substitution to construct a "bad" translation
bad_pro_trans_map = {
    "pleonastic_it": {"es": ["er", "sie"]},
    "anaphoric_intra_subject_it": {"es": ["er", "sie"], "er": ["es", "sie"], "sie": ["es", "er"]},
    "anaphoric_intra_non-subject_it": {"ihn": ["es", "sie"], "es": ["ihn", "sie"], "sie": ["es", "ihn"]},
    "anaphoric_intra_they": {"sie": ["es", "er"]},
    "anaphoric_singular_they": {"sie": ["es", "er"]},
    "anaphoric_group_it-they": {"es": ["er", "sie"], "er": ["es", "sie"], "sie": ["es", "er"]}
}


def get_pronoun_examples(file_name):
    """
    Read pronoun examples from the .csv file extracted from a PROTEST-style database
    :param file_name: .csv file output from PROTEST-style database
    :return: a list of examples in the ACES-format
    """
    examples = []
    examples_read = {} # src_corpus: example_no
    with open(file_name, "r") as f:
        csv_reader = reader(f)
        next(csv_reader, None)  # Skip the file header
        for row in csv_reader:
            srccorpus = row[6]
            example_no = row[8]
            if srccorpus in examples_read and example_no in examples_read[srccorpus]:
                # Don't read in the same example twice, but with different annotator judgements
                continue
            examples_read[srccorpus] = example_no
            try:
                pro_trans_position = int(row[13])
            except:
                # Skip the example if the pronoun was dropped from the MT translation
                continue
            pro_judgement = row[4]
            pro_type = row[16]
            source = row[17]
            reference = row[19]
            pro_trans_ref = get_reference_pro_trans(reference, int(row[14]))
            good_translation = row[18]
            bad_translation_deletion = get_bad_translation_deletion(good_translation, pro_trans_position)
            example_del = [source, good_translation, bad_translation_deletion, reference, pro_type+":deletion"]
            examples.append(example_del)
            if pro_judgement != "ok":
                # Only if the pronoun translation was marked as correct should we generate a substitution translation
                continue
            bad_translation_substitution, subs_trans = get_bad_translation_substitution(good_translation, pro_type, pro_trans_position)
            label = pro_type + ":substitution"
            if subs_trans.lower() == pro_trans_ref.lower():
                label += "_bad_pro_trans_same_as_ref"
            else:
                label += "_bad_pro_trans_different_to_ref"
            if bad_translation_substitution is not None:
                example_sub = [source, good_translation, bad_translation_substitution, reference, label]
                examples.append(example_sub)
    return examples


def get_bad_translation_substitution(mt_translation, pro_type, position):
    """
    Given a sentence, replace a "good" pronoun translation with a "bad" one from bad_pro_trans_map
    :param mt_translation: MT output string
    :param pro_type: type of pronoun, e.g. anaphoric, pleonastic, etc.
    :param position: position of the pronoun token (assumes a tokenised mt_translation)
    :return: bad translation and the substitution
    """
    # Tokenise the MT output
    mt_trans_toks = mt_translation.split(" ")
    # Get "good" pronoun translation
    pro_tgt_trans = mt_trans_toks[position]
    if pro_tgt_trans.lower() not in bad_pro_trans_map[pro_type]:
        return None, pro_tgt_trans
    # Select a "bad" alternative pronoun translation and replace the token in the sentence with it
    substitution = random.choice(bad_pro_trans_map[pro_type][pro_tgt_trans.lower()])
    if position == 0:
        substitution = substitution.capitalize()
    mt_trans_toks[position] = substitution
    bad_translation = " ".join(mt_trans_toks)
    return bad_translation, substitution


def get_bad_translation_deletion(mt_translation, position):
    """
    Given a sentence, delete a pronoun translation from bad_pro_trans_map
    :param mt_translation: MT output string
    :param position: position of the pronoun token (assumes a tokenised mt_translation)
    :return: bad translation
    """
    mt_trans_toks = mt_translation.split(" ")
    del mt_trans_toks[position]
    bad_trans = " ".join(mt_trans_toks)
    bad_trans = bad_trans.replace("  ", " ")
    return bad_trans


def get_reference_pro_trans(reference, position):
    """
    Given a reference translation (string) and a token position, return the pronoun translation from the reference
    :param reference: reference translation string
    :param position: position of the pronoun token (assumes a tokenised mt_translation)
    :return: the pronoun translation from the reference
    """
    ref_toks = reference.split(" ")
    pro_trans = ref_toks[position]
    return pro_trans


def write_output(file_name, examples):
    """
    Write output to file
    :param file_name: output file name
    :param examples: original input plus "incorrect" translation and phenomenon label
    :return: N/A
    """
    with open(file_name, "w") as o:
        header = ["source", "good-translation", "incorrect-translation", "reference", "phenomena"]
        header_line = "\t".join(header)
        o.write(header_line + "\n")
        for example in examples:
            example_line = "\t".join(example)
            o.write(example_line + "\n")


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-file", help=".csv file containing extraction from PROTEST database")
    parser.add_argument("-o", "--out-file", help=".tsv file containing the formatted examples")
    args = parser.parse_args()

    # Get examples from the PROTEST .csv file
    examples = get_pronoun_examples(args.in_file)

    # Write examples to .tsv file
    write_output(args.out_file, examples)