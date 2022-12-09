#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import string
import csv

random.seed(42)


def read_input_file(file_name):
    """
    Read contents of a tsv format input file to a list
    :param file_name: tsv format input file containing translations to be perturbed
    :return: list of examples, where an example looks like ["source", "good-translation", "", "reference", ""]
    """
    examples = []
    with open(file_name) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        next(tsvreader, None)  # Skip the headers
        for row in tsvreader:
            example = [row[0], row[1], "", row[2], ""]
            examples.append(example)
    return examples


def strip_all_punct(sentence):
    """
    Strip all punctuation marks from the sentence
    :param sentence: "good" translation
    :return: "incorrect" translation with all punctuation marks stripped out
    """
    stripped = sentence.translate(str.maketrans("", "", string.punctuation))
    return stripped


def strip_only_commas(sentence):
    """
    Strip all commas from the sentence
    :param sentence: "good" translation
    :return: "incorrect" translation with commas stripped out
    """
    stripped = sentence.translate(str.maketrans("", "", ","))
    return stripped


def strip_only_quotes(sentence):
    """
    Strip all double quotes from the sentence
    :param sentence: "good" translation
    :return: "incorrect" translation with quotes stripped out
    """
    stripped = sentence.translate(str.maketrans("", "", "\""))
    return stripped


def questionify(sentence):
    """
    Convert the sentence from a statement to a question
    :param sentence: "good" translation
    :return: "incorrect" translation with punctuation replacement applied (! -> ?)
    """
    stripped = sentence.replace("!", "?")
    return stripped


def get_permutation_type(sentence):
    """
    Determine which type of perturbation to apply given the punctuation marks present in the "good" translation
    :param sentence: "good" translation
    :return: perturbation function name (to be applied) and corresponding phenomenon label
    """
    if "!" in sentence:
        return questionify, "punctuation:statement-to-question"
    elif "\"" in sentence:
        return strip_only_quotes, "punctuation:deletion_quotes"
    else:
        if "," in sentence:
            # Flip a coin between stripping out punctuation / replacing commas
            options = [ [strip_only_commas,"punctuation:deletion_commas"],
                        [strip_all_punct, "punctuation:deletion_all"] ]
            pt = random.choice(options)
            return pt[0], pt[1]
        else:
            return strip_all_punct, "punctuation:deletion_all"


def permute_punctuation(examples):
    """
    Select and perform perturbations
    :param examples:
    :return:
    """
    for i in range(0, len(examples)):
        good_trans = examples[i][1]
        func, label = get_permutation_type(good_trans)
        bad_trans = func(good_trans)
        examples[i][2] = bad_trans
        examples[i][4] = label
    return examples


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
    parser.add_argument("-i", "--in-file", help=".tsv file with source/reference/translation")
    parser.add_argument("-o", "--out-file", help=".tsv file with perturbations and phenomenon labels")
    args = parser.parse_args()

    # Read examples from file
    examples = read_input_file(args.in_file)

    # Perform perturbations
    examples = permute_punctuation(examples)

    # Write output file
    write_output(args.out_file, examples)
