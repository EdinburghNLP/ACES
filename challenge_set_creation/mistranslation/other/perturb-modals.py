#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import csv
from googletrans import Translator
from tqdm import tqdm
from nltk.tokenize import word_tokenize

random.seed(42)

english_modal_verbs = ["must", "may", "might", "could"]


def read_input_file(file_name):
    """
    Read contents of a tsv format input file to a list
    :param file_name: tsv format input file containing translations to be perturbed
    :return: two lists of sentences: en_sentences and de_sentences
    """
    en_sentences = {}
    de_sentences = {}
    with open(file_name) as f:
        tsvreader = csv.reader(f, delimiter="\t")
        next(tsvreader, None)  # Skip the headers
        for i, row in enumerate(tsvreader):
            en_sentences[i] = row[1]
            de_sentences[i] = row[0]
    return en_sentences, de_sentences


def translate(sentence, src, tgt):
    """
    Translate a single sentence using Google's translation API
    :param sentence: input sentence (in source language)
    :param src: source language
    :param tgt: target language
    :return: target-language translation
    """
    try:
        translation = translator.translate(sentence, src=src, dest=tgt)
    except:
        print()
    return translation.text


def translate_sentences(sentences, src, tgt):
    """
    Bulk translation - call "translate" for each sentence in turn
    :param sentences: dictionary of sentences
    :param src: source language
    :param tgt: target language
    :return: dictionary of sentences with MT output added
    """
    for sent_num, v in tqdm(sentences.items()):
        src_sentence = sentences[sent_num]["source"]
        mt_output = translate(src_sentence, src, tgt)
        sentences[sent_num]["mt_output"] = mt_output
    return sentences


def filter_modal_verbs(en_sentences, de_sentences):
    """
    Extract de-en sentence pairs where the en reference sentence contains a modal verb from the list english_modal_verbs
    :param en_sentences: en sentences (list)
    :param de_sentences: de sentences (list)
    :return: filtered sentence pairs
    """
    filtered_sentences = {}
    for sent_num in en_sentences:
        en_sentence = en_sentences[sent_num]
        count_modal_verbs = 0
        for modal_verb in english_modal_verbs:
            count_modal_verbs += en_sentence.split().count(modal_verb)
        # For simplicity, consider only sentences with a single (en) modal verb
        if count_modal_verbs == 1:
            de_sentence = de_sentences[sent_num]
            # Filter out sentences where double quotes are present in the source / reference
            if not "\"" in de_sentence and not "\"" in en_sentence \
                and not "“" in de_sentence and not "“" in en_sentence:
                filtered_sentences[sent_num] = {"reference": en_sentence,
                                                "source": de_sentence}
    return filtered_sentences


def write_output(file_name, examples):
    """
    Write output to file
    :param file_name: output file name
    :param examples: original input plus "incorrect" translation and phenomenon label
    :return: N/A
    """
    header_cols = ["source", "good-translation", "incorrect-translation", "reference", "phenomena"]
    with open(file_name, "w") as o:
        header = "\t".join(header_cols)
        o.write(header + "\n")
        for example in examples:
            example_line = "\t".join(example)
            o.write(example_line + "\n")


def construct_modal_examples(sentences):
    """
    Construct new modal examples, by translating and perturbing MT output
    :param sentences: dictionary of sentences
    :return: examples
    """
    exs = []
    for sent_num in sentences:
        mt_output = sentences[sent_num]["mt_output"]
        source = sentences[sent_num]["source"]
        reference = sentences[sent_num]["reference"]

        # Tokenise the MT output
        mt_output_tokenised = word_tokenize(mt_output)

        modal_of_interest = False

        for mv in english_modal_verbs:
            # Check if there is a modal verb in the MT output or the reference
            if mv in mt_output_tokenised or mv in reference:
                modal_of_interest = True
                # Deletion - good trans to be VERIFIED manually
                mt_output_deletion = mt_output
                mt_output_deletion = mt_output_deletion.replace(mv, "").replace("  ", " ")
                exs.append([source, mt_output, mt_output_deletion, reference, "modal_verb:deletion"])
                # Substitution - good and bad trans to be COMPLETED manually
                exs.append([source, mt_output, mt_output, reference, "modal_verb:substitution"])
    return exs


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in-file", help="Input file containing parallel de-en sentences (.tsv format)")
    parser.add_argument("-o", "--out-file", help="Output file in .tsv format")
    args = parser.parse_args()

    # Set up a translator
    translator = Translator()

    # Read parallel sentences
    en_sentences, de_sentences = read_input_file(args.in_file)

    # Find English sentences with a single modal verb (and the German aligned sentence)
    modal_verb_sentences = filter_modal_verbs(en_sentences, de_sentences)

    # Translate each German sentence to English
    modal_verb_sentences = translate_sentences(modal_verb_sentences, 'de', 'en')

    # Construct examples
    examples = construct_modal_examples(modal_verb_sentences)

    # Write output
    write_output(args.out_file, examples)