#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datasets import load_from_disk

import numpy as np
np.random.seed(42)

import json, copy, os, argparse
from tqdm import tqdm

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from annotation_utilities import *
from debugging_utilities import *

# process given sample, annotate or do manual annotation (only in the annotations.ipynb, in process_dataset.py only automatic annotation)
def process_sample(idx, sample, manual=False, detokenize=False):
    if phenomena[sample["phenomena"]] == 'mixed_flexible':
        good_og = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"])
    elif phenomena[sample["phenomena"]] == 'REF_flexible':
        good_og = sample["reference"]
    else:
        good_og = sample["good-translation"]
    bad_og = sample["incorrect-translation"]
    # if detokenize we just annotate the detokenized sentences, then map the character span back to the original sentence
    # in the standardize_annotation function in annotation_utilities.py
    if detokenize:
        try:
            good, good_map = detokenize_text(good_og, lang=sample["langpair"].split('-')[1])
            bad, bad_map = detokenize_text(bad_og, lang=sample["langpair"].split('-')[1])
            maps = (good_map, bad_map)
        except:
            good, bad = good_og, bad_og
            maps = None
    else:
        good, bad = good_og, bad_og
        maps = None # the standardize_annotation function will understand that it does not need to revert detokenization 
        # if maps parameter is None.
    originals = (good_og, bad_og)
    # tokenize here, so we can check if all the annotations are word level later
    if good.lower() != bad.lower():
        g, g_spans = tokenize(good.lower())
        b, b_spans = tokenize(bad.lower())
    else:
        g, g_spans = tokenize(good)
        b, b_spans = tokenize(bad)
    if phenomena[sample["phenomena"]] == 'add-omit':
        try:
            change = diff_char_level(good, bad)
            if len(change) == 0:
                logger.warning('No change in id {}'.format(idx))
            else:
                change = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample
        except:
            logger.warning('error in char level annotate, id {}'.format(idx))
        stats[sample["phenomena"]]["total"] += 1

    elif phenomena[sample["phenomena"]] == 'annotate_word':
        try:
            change = annotate_word(good, bad)
            if len(change) == 0:
                logger.warning('No change in id {}'.format(idx))
                stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
            else:
                stats[sample["phenomena"]]["success"] += 1
                assert is_word_level(g_spans, b_spans, change)
                change = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample
        except:
            logger.warning('error in word level annotate, id {}'.format(idx))
            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
        stats[sample["phenomena"]]["total"] += 1

    elif phenomena[sample["phenomena"]] in ['diff_flexible', 'REF_flexible', 'mixed_flexible']:
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
                assert is_word_level(g_spans, b_spans, change)
                change = standardize_annotation(change, good, bad, maps, originals)
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
                    assert is_word_level(g_spans, b_spans, change)
                    change = standardize_annotation(change, good, bad, maps, originals)
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample
            except: 
                logger.warning('error in id {}'.format(idx))
                stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
        stats[sample["phenomena"]]["total"] += 1
        
    elif phenomena[sample["phenomena"]] == 'units':
        try:
            g, b, change = annotate_units(good,bad)
            if len(change) == 0 and g != b:
                logger.warning('No change in id {}, \ng: {}, \nb: {},\nr: {}'.format(idx, g, b))
                stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
            elif len(change) > 1:
                logger.warning('Multiple changes in {} id {}'.format(sample["phenomena"], idx))
                stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))
            else:
                stats[sample["phenomena"]]["success"] += 1
                assert is_word_level(g_spans, b_spans, change)
                change = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample  
        except: 
            logger.warning('error in id {}'.format(idx))
            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
        stats[sample["phenomena"]]["total"] += 1

    elif phenomena[sample["phenomena"]] == 'swap':
        try:
            change = annotate_swap_word_lvl(good,bad)
            if len(change) < 2 and good != bad:
                logger.warning('No change in id {}, \ng: {}, \nb: {}'.format(idx, good, bad))
                stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
            elif change[0]['in_good'] != None and change[1]['in_good'] != None and change[0]['in_good'] == change[1]['in_good']:
                logger.warning('check this: %s - swapped words are the same!' %idx)
                stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))
            elif (change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 50) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 50):
                logger.warning('check this: %s' %idx)
                stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
            else:
                stats[sample["phenomena"]]["success"] += 1
                assert is_word_level(g_spans, b_spans, change)
                change = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample
        except: 
            logger.warning('error in id {}'.format(idx))
            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
        stats[sample["phenomena"]]["total"] += 1

    elif phenomena[sample["phenomena"]] == 'date':
        try:
            change = diff_dates(good,bad)
            stats[sample["phenomena"]]["success"] += 1
            assert is_word_level(g_spans, b_spans, change)
            change = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample
        except: 
            logger.warning('error in id {}'.format(idx))
            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
        stats[sample["phenomena"]]["total"] += 1
            
    elif phenomena[sample['phenomena']] == 'whole_sentence':
        change = whole_sentence(good, bad)
        stats[sample["phenomena"]]["success"] += 1
        assert is_word_level(g_spans, b_spans, change)
        change = standardize_annotation(change, good, bad, maps, originals)
        sample['annotation'] = change
        sample['method'] = phenomena[sample["phenomena"]]
        annotations[idx] = sample
        stats[sample["phenomena"]]["total"] += 1
    
    return 1  # 1 for success
        
def process_phenomena(samples, manual=False, detokenize=False):
    for idx,sample in tqdm(samples.items()):
        if idx not in annotations.keys() and int(idx) not in annotations.keys():            
            # check if it was annotated before
            res = check_seen_before(sample, annotations)
            if res != None:
                sample['annotation'] = res[0][0]
                sample['method'] = res[0][1] + " - duplicate of: " + str(res[1])
                annotations[int(idx)] = sample
                stats[sample["phenomena"]]["duplicate"].append(idx)
                stats[sample["phenomena"]]["total"] += 1
                stats[sample["phenomena"]]["success"] += 1
            else:
                # if sample["phenomena"] == "ambiguous-translation-wrong-discourse-connective-since-causal":
                res = process_sample(idx, sample, manual, detokenize)

                if res == -1:
                    return -1

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--detokenize",default=False, required=bool, help="the sentences will be detokenized, then annotated, then will be mapped back to original")
    # parser.add_argument("-c", "--checkpoint",default=False, required=bool, help="load annotated.txt and stats.txt from checkpoint (True or False)")
    args = parser.parse_args()

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
        'addition':'annotate_word',
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
        'copy-source':'whole_sentence',
        'coreference-based-on-commonsense':'manual',
        'do-not-translate':'whole_sentence',
        'hallucination-date-time':'date',
        'hallucination-named-entity-level-1':'diff_flexible',
        'hallucination-named-entity-level-2':'REF_flexible',
        'hallucination-named-entity-level-3':'REF_flexible',
        'hallucination-number-level-1':'diff_flexible',
        'hallucination-number-level-2':'REF_flexible',
        'hallucination-number-level-3':'REF_flexible',
        'hallucination-real-data-vs-ref-word':'manual',
        'hallucination-real-data-vs-synonym':'manual',
        'hallucination-unit-conversion-amount-matches-ref':'units',
        'hallucination-unit-conversion-unit-matches-ref':'units',
        'hypernym-replacement':'REF_flexible',
        'hyponym-replacement':'REF_flexible',
        'lexical-overlap':'manual',
        'modal_verb:deletion':'add-omit',
        'modal_verb:substitution':'diff_flexible',
        'nonsense':'REF_flexible',
        'omission':'annotate_word',
        'ordering-mismatch':'swap',
        'overly-literal-vs-correct-idiom':'diff_flexible',
        'overly-literal-vs-explanation':'diff_flexible',
        'overly-literal-vs-ref-word':'diff_flexible',
        'overly-literal-vs-synonym':'diff_flexible',
        'pleonastic_it:deletion':'annotate_word',
        'pleonastic_it:substitution':'annotate_word',
        'punctuation:deletion_all':'annotate_word',
        'punctuation:deletion_commas':'annotate_word',
        'punctuation:deletion_quotes':'annotate_word',
        'punctuation:statement-to-question':'annotate_word',
        'real-world-knowledge-entailment':'diff_flexible',
        'real-world-knowledge-hypernym-vs-distractor':'diff_flexible',
        'real-world-knowledge-hypernym-vs-hyponym':'diff_flexible',
        'real-world-knowledge-synonym-vs-antonym':'diff_flexible',
        'similar-language-high':'whole_sentence',
        'similar-language-low':'whole_sentence',
        'untranslated-vs-ref-word':'diff_flexible',   # here add-omit can be used for getting character level replacements too
        'untranslated-vs-synonym':'diff_flexible',
        'xnli-addition-contradiction':'canceled',
        'xnli-addition-neutral':'canceled',
        'xnli-omission-contradiction':'canceled',
        'xnli-omission-neutral':'canceled'
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
        annotations = {int(k):v for k,v in annotations.items()}
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
                'duplicate':[],
                'not_word_level':[],
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
        for p in phenomena_tobe_processed:
            stats[p] = copy.deepcopy(stats_template)

    logger.info('Processing running... detokenize: {}'.format(args.detokenize))
    logger.setLevel(logging.ERROR)
    
    # to completely reset the prev annotations
    annotations = dict()
    stats = {}
    for key in phenomena_tobe_processed:
        stats[key] = copy.deepcopy(stats_template)
    samples = dict()
    for idx, sample in enumerate(dataset['train']):
        if sample['phenomena'] in phenomena_tobe_processed:
            samples[idx] = sample    
    process_phenomena(samples, manual=False, detokenize=args.detokenize)
    
    with open(annotated_dataset_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)  # encode dict into JSON
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)  # encode dict into JSON
    logger.info("Done writing dict into {} file".format(annotated_dataset_path))