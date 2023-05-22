#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datasets import load_from_disk

import numpy as np
np.random.seed(42)

import json, copy, os
from tqdm import tqdm

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from annotation_utilities import *

folder = os.getcwd()
dataset_path = os.path.join(folder, 'dataset')
if not os.path.exists(dataset_path):
    logger.error('No dataset path: %s' %(dataset_path))
    exit()

logger.info('Loading the dataset...')
dataset = load_from_disk(dataset_path)
logger.info('Dataset loaded.')

# this is the list of phenomena and which option they need to be annotated with:
phenomena = {
    'addition':'add-omit',
    'ambiguous-translation-wrong-discourse-connective-since-causal':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-since-temporal':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-while-contrast':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-while-temporal':'diff_flexible',
    'ambiguous-translation-wrong-gender-female-anti':'diff_flexible',
    'ambiguous-translation-wrong-gender-female-pro':'diff_flexible',
    'ambiguous-translation-wrong-gender-male-anti':'diff_flexible',
    'ambiguous-translation-wrong-gender-male-pro':'diff_flexible',
    'ambiguous-translation-wrong-sense-frequent':'diff_flexible',
    'ambiguous-translation-wrong-sense-infrequent':'diff_flexible',
    'anaphoric_group_it-they:deletion':'annotate_word',
    'anaphoric_group_it-they:substitution':'annotate_word',
    'anaphoric_intra_non-subject_it:deletion':'annotate_word',
    'anaphoric_intra_non-subject_it:substitution':'annotate_word',
    'anaphoric_intra_subject_it:deletion':'annotate_word',
    'anaphoric_intra_subject_it:substitution':'annotate_word',
    'anaphoric_intra_they:deletion':'annotate_word',
    'anaphoric_intra_they:substitution':'annotate_word',
    'anaphoric_singular_they:deletion':'annotate_word',
    'anaphoric_singular_they:substitution':'annotate_word',
    'antonym-replacement':'REF_flexible',
    'commonsense-only-ref-ambiguous':'diff_flexible',
    'commonsense-src-and-ref-ambiguous':'diff_flexible',
    'copy-source':'?',
    'coreference-based-on-commonsense':'mixed_flexible',
    'do-not-translate':'diff_flexible',
    'hallucination-date-time':'date',
    'hallucination-named-entity-level-1':'diff_flexible',
    'hallucination-named-entity-level-2':'REF_flexible',
    'hallucination-named-entity-level-3':'REF_flexible',
    'hallucination-number-level-1':'diff_flexible',
    'hallucination-number-level-2':'REF_flexible',
    'hallucination-number-level-3':'REF_flexible',
    'hallucination-real-data-vs-ref-word':'diff_flexible',
    'hallucination-real-data-vs-synonym':'diff_flexible',
    'hallucination-unit-conversion-amount-matches-ref':'units',
    'hallucination-unit-conversion-unit-matches-ref':'units',
    'hypernym-replacement':'REF_flexible',
    'hyponym-replacement':'REF_flexible',
    'lexical-overlap':'?',
    'modal_verb:deletion':'add-omit',
    'modal_verb:substitution':'diff_flexible',
    'nonsense':'REF_flexible',
    'omission':'add-omit',
    'ordering-mismatch':'swap',
    'overly-literal-vs-correct-idiom':'diff_flexible',
    'overly-literal-vs-explanation':'diff_flexible',
    'overly-literal-vs-ref-word':'diff_flexible',
    'overly-literal-vs-synonym':'diff_flexible',
    'pleonastic_it:deletion':'annotate_word',
    'pleonastic_it:substitution':'annotate_word',
    'punctuation:deletion_all':'add-omit',
    'punctuation:deletion_commas':'add-omit',
    'punctuation:deletion_quotes':'add-omit',
    'punctuation:statement-to-question':'add-omit',
    'real-world-knowledge-entailment':'diff_flexible',
    'real-world-knowledge-hypernym-vs-distractor':'diff_flexible',
    'real-world-knowledge-hypernym-vs-hyponym':'diff_flexible',
    'real-world-knowledge-synonym-vs-antonym':'diff_flexible',
    'similar-language-high':'?',
    'similar-language-low':'?',
    'untranslated-vs-ref-word':'diff_flexible',   # here add-omit can be used for getting character level replacements too
    'untranslated-vs-synonym':'diff_flexible',
    'xnli-addition-contradiction':'?',
    'xnli-addition-neutral':'?',
    'xnli-omission-contradiction':'?',
    'xnli-omission-neutral':'?'
}

# change this part to specify which phenomena to process!
"""
phenomena_tobe_processed = ["coreference-based-on-commonsense", "hallucination-real-data-vs-ref-word",
                            "hallucination-real-data-vs-synonym", "lexical-overlap", "xnli-addition-contradiction",
                            "xnli-addition-neutral", "xnli-omission-contradiction", "xnli-omission-neutral"]
"""
# or to process all of them
phenomena_tobe_processed = phenomena.keys()

# if there are already some annotations overwrite them and append new ones
annotated_dataset_path = os.path.join(folder, 'ACES_private/challenge_set_annotation/annotated.txt')
if os.path.exists(annotated_dataset_path):
    logger.info('Path {} already exists. Loading..'.format(annotated_dataset_path))
    with open(annotated_dataset_path, "r") as f:
        annotations = json.load(f)
else:
    logger.info('Creating new annotations.txt file at {}'.format(annotated_dataset_path))
    annotations = dict()

# calculate statistics about the annotations:
# for every mode, calculate no. of skipped, no. of unsure and ids, and no. of done.
stats_template = {
            'total':0,
            'success':0,
            'too_long':[],
            'no_change':[],
            'error':[],
            'other':[]  
        }
stats_path = os.path.join(folder, 'ACES_private/challenge_set_annotation/stats.txt')
if os.path.exists(stats_path):
    logger.info('Path {} already exists. Loading..'.format(stats_path))
    with open(stats_path, "r") as f:
        stats = json.load(f)
    # we want to overwrite the statistics for the new phenomena
    for p in phenomena_tobe_processed:
        stats[p] = copy.deepcopy(stats_template)
else:
    logger.info('Creating new stats.txt file at {}'.format(stats_path))
    stats = {}
    for key in phenomena.keys():
        stats[key] = copy.deepcopy(stats_template)

logger.info('Processing running...')
logger.setLevel(logging.ERROR)
for idx,sample in tqdm(enumerate(dataset["train"])):
    # if sample["phenomena"] in phenomena.keys() and 'Le garçon voulait mettre le jeu dans la boîte, mais c\'était trop grand.' in sample['good-translation']:
    if sample["phenomena"] in phenomena_tobe_processed:
        try:
            stats[sample["phenomena"]]["total"] += 1
            if phenomena[sample["phenomena"]] == 'add-omit':
                try:
                    change = diff_char_level(sample["good-translation"], sample["incorrect-translation"])
                    if len(change) == 0:
                        logger.warning('No change in id {}'.format(idx))
                        stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                    else:
                        stats[sample["phenomena"]]["success"] += 1
                        change = standardize_annotation(change, sample["good-translation"], sample["incorrect-translation"])
                    sample['annotation'] = change
                    sample['method'] = phenomena[sample["phenomena"]]
                    annotations[idx] = sample
                except:
                    logger.warning('error in char level annotate, id {}'.format(idx))
                    stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

            elif phenomena[sample["phenomena"]] == 'annotate_word':
                try:
                    change = annotate_word(sample["good-translation"], sample["incorrect-translation"])
                    if len(change) == 0:
                        logger.warning('No change in id {}'.format(idx))
                        stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                    else:
                        stats[sample["phenomena"]]["success"] += 1
                        change = standardize_annotation(change, sample["good-translation"], sample["incorrect-translation"])
                    sample['annotation'] = change
                    sample['method'] = phenomena[sample["phenomena"]]
                    annotations[idx] = sample
                except:
                    logger.warning('error in word level annotate, id {}'.format(idx))
                    stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

            elif phenomena[sample["phenomena"]] in ['diff_flexible', 'REF_flexible', 'mixed_flexible']:
                if phenomena[sample["phenomena"]] == 'diff_flexible':
                    good = sample["good-translation"]
                elif phenomena[sample["phenomena"]] == 'mixed_flexible':
                    good, g, g_spans = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"])
                else: 
                    good = sample["reference"]
                bad = sample["incorrect-translation"]
                g, g_spans = tokenize(good)
                b, b_spans = tokenize(bad)

                # special treatment to japanese chinese and thailandish because they don't use spaces, so can't be split            
                if sample['langpair'][-2:] not in ['ja', 'zh', 'th']:      
                    if len(g) == len(b):   # if there are multiple one word replacements
                        change = diff(g, g_spans, b, b_spans, phenomena="replacement")
                    if len(g) != len(b) or len(change) == 0:
                        try:
                            change = diff_flexible(good, g, g_spans, bad, b, b_spans)
                            if len(change) == 0 and good != bad:
                                change = diff_char_level(good, bad) 
                        except:
                            logger.warning('error in id {}'.format(idx))
                            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
                    if len(change) == 0:
                        logger.warning('No change in id {}'.format(idx,g,b,change))
                        stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                    elif len(change) != 0 and ((change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 50) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 50)):
                        logger.warning('check this - too long: %s' %idx)
                        stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
                    else:
                        stats[sample["phenomena"]]["success"] += 1
                        change = standardize_annotation(change, good, bad)
                    sample['annotation'] = change
                    sample['method'] = phenomena[sample["phenomena"]]
                    annotations[idx] = sample  
                else:
                    try:
                        change = diff_char_level(good, bad) 
                        if len(change) == 0 and good != bad:
                            logger.warning('No change in id {}'.format(idx,g,b,change))
                            stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                        elif len(change) != 0 and ((change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 30) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 30)):
                            logger.warning('check this - too long: %s' %idx)
                            stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
                        else:
                            stats[sample["phenomena"]]["success"] += 1
                            change = standardize_annotation(change, good, bad)
                        sample['annotation'] = change
                        sample['method'] = phenomena[sample["phenomena"]]
                        annotations[idx] = sample
                    except: 
                        logger.warning('error in id {}'.format(idx))
                        stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

            elif phenomena[sample["phenomena"]] == 'units':
                try:
                    g, b, change = annotate_units(sample["good-translation"],sample["incorrect-translation"])
                    if len(change) == 0 and g != b:
                        logger.warning('No change in id {}, \ng: {}, \nb: {},\nr: {}'.format(idx, g, b))
                        stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                    elif len(change) > 1:
                        logger.warning('Multiple changes in {} id {}'.format(sample["phenomena"], idx))
                        stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))
                    else:
                        stats[sample["phenomena"]]["success"] += 1
                        change = standardize_annotation(change, sample["good-translation"], sample["incorrect-translation"])
                    sample['annotation'] = change
                    sample['method'] = phenomena[sample["phenomena"]]
                    annotations[idx] = sample  
                except: 
                    logger.warning('error in id {}'.format(idx))
                    stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

            elif phenomena[sample["phenomena"]] == 'swap':
                try:
                    change = annotate_swap_word_lvl(sample["good-translation"],sample["incorrect-translation"])
                    if len(change) < 2 and sample["good-translation"] != sample["incorrect-translation"]:
                        logger.warning('No change in id {}, \ng: {}, \nb: {}'.format(idx, sample["good-translation"], sample["incorrect-translation"]))
                        stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                    elif change[0]['in_good'] != None and change[1]['in_good'] != None and change[0]['in_good'] == change[1]['in_good']:
                        logger.warning('check this: %s - swapped words are the same!' %idx)
                        stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))
                    elif (change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 50) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 50):
                        logger.warning('check this: %s' %idx)
                        stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
                    else:
                        stats[sample["phenomena"]]["success"] += 1
                        change = standardize_annotation(change, sample["good-translation"], sample["incorrect-translation"])
                    sample['annotation'] = change
                    sample['method'] = phenomena[sample["phenomena"]]
                    annotations[idx] = sample
                except: 
                    logger.warning('error in id {}'.format(idx))
                    stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

            elif phenomena[sample["phenomena"]] == 'date':
                try:
                    change = diff_dates(sample["good-translation"],sample["incorrect-translation"])
                    stats[sample["phenomena"]]["success"] += 1
                    change = standardize_annotation(change, sample["good-translation"], sample["incorrect-translation"])
                    sample['annotation'] = change
                    sample['method'] = phenomena[sample["phenomena"]]
                    annotations[idx] = sample
                except: 
                    logger.warning('error in id {}'.format(idx))
                    stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
            elif phenomena[sample['phenomena']] == 'whole_sentence':
                change = whole_sentence(sample["good-translation"], sample["incorrect-translation"])
                stats[sample["phenomena"]]["success"] += 1
                change = standardize_annotation(change, sample["good-translation"], sample["incorrect-translation"])
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample
        except:
            logger.error("Problem!! id: {}".format(idx))
with open(annotated_dataset_path, "w") as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)  # encode dict into JSON
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)  # encode dict into JSON
logger.info("Done writing dict into {} file".format(annotated_dataset_path))