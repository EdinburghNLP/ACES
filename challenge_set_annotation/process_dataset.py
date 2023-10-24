#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from datasets import load_from_disk

import numpy as np
np.random.seed(42)

import json, copy, os, sys, argparse, pandas
from tqdm import tqdm
from collections import defaultdict

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from annotation_utilities import *
from debugging_utilities import *

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "ACES_private/span_predictions")))
from format_utilities import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "ACES_private/aces")))

# process given sample, annotate or do manual annotation (only in the annotations.ipynb, in process_dataset.py only automatic annotation)
def process_sample(idx, sample, manual=False, detokenize=False):
    if phenomena[sample["phenomena"]] in ['manual', 'canceled'] or str(idx) in manual_ids:
        return 1
    if sample['langpair'][-2:] in ['ja', 'zh', 'th', 'ko']:  
        chars = True
    else:
        chars = False
    if phenomena[sample["phenomena"]] == 'mixed_flexible':
        good_og = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"], chars=chars)
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
        g, g_spans = tokenize(good.lower(), chars=chars)
        b, b_spans = tokenize(bad.lower(), chars=chars)
    else:
        g, g_spans = tokenize(good, chars=chars)
        b, b_spans = tokenize(bad, chars=chars)
        
    if phenomena[sample["phenomena"]] == 'annotate_word':
        stats[sample["phenomena"]]["total"] += 1
        try:
            change = annotate_word(good, bad, chars=chars)
            if len(change) == 0:
                logger.warning('No change in id {}'.format(idx))
                stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
            else:
                assert is_word_level(g_spans, b_spans, change)
                stats[sample["phenomena"]]["success"] += 1
                change, omission = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['omission'] = omission
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample
        except:
            logger.warning('error in word level annotate, id {}'.format(idx))
            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

    elif phenomena[sample["phenomena"]] in ['diff_flexible', 'REF_flexible', 'mixed_flexible']:
        stats[sample["phenomena"]]["total"] += 1
        # special treatment to japanese chinese and thailandish because they don't use spaces, so can't be split            
        if sample['langpair'][-2:] not in ['ja', 'zh', 'th', 'ko']:      
            if len(g) == len(b):   # if there are multiple one word replacements
                change = diff(g, g_spans, b, b_spans, phenomena="replacement")
            if len(g) != len(b) or len(change) == 0:
                try:
                    change = diff_flexible(good, g, g_spans, bad, b, b_spans)
                    if len(change) == 0 and good != bad:
                        change = diff_char_level(good, bad) 
                        if len(change) == 0 and good != bad:
                            logger.warning('No change in id {}'.format(idx))
                            stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                            return 0
                except:
                    logger.warning('error in id {}'.format(idx))
                    stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
                    return 0
            if good != bad and ((change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 50) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 50)):
                logger.warning('check this - too long: %s' %idx)
                stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample  
                return 0
            assert is_word_level(g_spans, b_spans, change)
            stats[sample["phenomena"]]["success"] += 1     
            change, omission = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['omission'] = omission
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample  
        else:
            try:
                g, g_spans = tokenize(good.lower(), chars=True)
                b, b_spans = tokenize(bad.lower(), chars=True)
                if len(g) == len(b):   # if there are multiple one word replacements
                    change = diff(g, g_spans, b, b_spans, phenomena="replacement")
                if len(g) != len(b) or len(change) == 0:
                    change = diff_flexible(good, g, g_spans, bad, b, b_spans)
                    if len(change) == 0 and good != bad:
                        change = diff_char_level(good, bad) 
                        if len(change) == 0 and good != bad:
                            logger.warning('No change in id {}'.format(idx))
                            stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                
                if len(change) != 0:
                    if (change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 50) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 50):
                        logger.warning('check this - too long: %s' %idx)
                        stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
                    else:
                        stats[sample["phenomena"]]["success"] += 1
                    assert is_word_level(g_spans, b_spans, change)
                change, omission = standardize_annotation(change, good, bad, maps, originals)
                sample['annotation'] = change
                sample['omission'] = omission
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample
            except: 
                logger.warning('error in id {}'.format(idx))
                stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
        
    elif phenomena[sample["phenomena"]] == 'units':
        stats[sample["phenomena"]]["total"] += 1
        try:
            g, b, change = annotate_units(good,bad, mode=sample["phenomena"])
            if len(change) == 0 and good != bad:
                logger.warning('No change in id {}'.format(idx))
                stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
            elif len(change) > 1:
                logger.warning('Multiple changes in {} id {}'.format(sample["phenomena"], idx))
                stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))
            else:
                stats[sample["phenomena"]]["success"] += 1
                assert is_word_level(g_spans, b_spans, change)
            change, omission = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['omission'] = omission
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample  
        except: 
            logger.warning('error in id {}'.format(idx))
            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

    elif phenomena[sample["phenomena"]] == 'swap':
        stats[sample["phenomena"]]["total"] += 1
        try:
            change = annotate_swap_word_lvl(good,bad)
            if len(change) < 2 and good != bad:
                logger.warning('No change in id {}'.format(idx))
                stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
            elif change[0]['in_good'] != None and change[1]['in_good'] != None and change[0]['in_good'] == change[1]['in_good']:
                logger.warning('check this: %s - swapped words are the same!' %idx)
                stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))
            elif (change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 50) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 50):
                logger.warning('too long: %s' %idx)
                stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
            else:
                stats[sample["phenomena"]]["success"] += 1
                assert is_word_level(g_spans, b_spans, change)
            change, omission = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['omission'] = omission
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample
        except: 
            logger.warning('error in id {}'.format(idx))
            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

    elif phenomena[sample["phenomena"]] == 'date':
        stats[sample["phenomena"]]["total"] += 1
        try:
            change = diff_dates(good,bad)
            stats[sample["phenomena"]]["success"] += 1
            assert is_word_level(g_spans, b_spans, change)
            change, omission = standardize_annotation(change, good, bad, maps, originals)
            sample['annotation'] = change
            sample['omission'] = omission
            sample['method'] = phenomena[sample["phenomena"]]
            annotations[idx] = sample
        except: 
            logger.warning('error in id {}'.format(idx))
            stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

    elif phenomena[sample['phenomena']] == 'whole_sentence':
        stats[sample["phenomena"]]["total"] += 1
        change = whole_sentence(good, bad)
        stats[sample["phenomena"]]["success"] += 1
        # assert is_word_level(g_spans, b_spans, change)
        change, omission = standardize_annotation(change, good, bad, maps, originals)
        sample['annotation'] = change
        sample['omission'] = omission
        sample['method'] = phenomena[sample["phenomena"]]
        annotations[idx] = sample

    try:
        assert len(sample['annotation']) == 0 or type(sample['annotation'][0]) == dict
    except:
        logger.warning("len(sample['annotation']) > 0 but not a dict: {}".format(idx))
        # return -1
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
                stats[sample["phenomena"]]["duplicate"] += 1
                stats[sample["phenomena"]]["total"] += 1
                stats[sample["phenomena"]]["success"] += 1
            else:
                # if sample["phenomena"] == "ambiguous-translation-wrong-discourse-connective-since-causal":
                res = process_sample(idx, sample, manual, detokenize)

                if res == -1:
                    return -1
                
def load_dataset(file=None):
    if file:
        dataset = {}
        with open(file, "r") as f:
            content = f.read()
        return load_tsv_dataset(content)
    folder = os.getcwd()
    dataset_path = os.path.join(folder, 'dataset')
    if not os.path.exists(dataset_path):
        logger.error('No dataset path: %s' %(dataset_path))
        exit()

    logger.info('Loading the dataset...')
    dataset = load_from_disk(dataset_path)
    logger.info('Dataset loaded.')
    return dataset["train"]

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-tsv", "--tsv_path",default=None, help="if the dataset is in tsv format dataset path.")
    parser.add_argument("-d", "--detokenize",default=False, required=bool, help="the sentences will be detokenized, then annotated, then will be mapped back to original")
    parser.add_argument("-c", "--checkpoint",default=False, help="load annotated.txt from given path")
    parser.add_argument("-r", "--reset", default=False, action='store_true', help="True for annotating the whole dataset from scratch")
    parser.add_argument("-debug", "--debug", default=False, action='store_true', help="True for using the logger in debugging mode")
    args = parser.parse_args()

    if args.tsv_path:
        dataset = load_dataset(args.tsv_path)
    else:
        dataset = load_dataset()
    if args.reset:
        annotated_dataset_path = ""
    elif args.checkpoint:
        if not os.path.exists(args.checkpoint):
            logger.error("The checkpoint path {} does not exists. Running on the whole dataset".format(args.checkpoint))
            # if there are already some annotations overwrite them and append new ones
            annotated_dataset_path = os.path.join(folder, 'ACES_private/challenge_set_annotation/annotated.txt')
        else:
            annotated_dataset_path = args.checkpoint
    else:
        annotated_dataset_path = os.path.join(folder, 'ACES_private/challenge_set_annotation/annotated.txt')

    # change this part to specify which phenomena to process!
    """
    phenomena_tobe_processed = ["coreference-based-on-commonsense", "hallucination-real-data-vs-ref-word",
                                "hallucination-real-data-vs-synonym", "lexical-overlap", "xnli-addition-contradiction",
                                "xnli-addition-neutral", "xnli-omission-contradiction", "xnli-omission-neutral"]
    """
    # or to process all of them
    phenomena_tobe_processed = phenomena.keys()

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
                'duplicate':0,
                'not_word_level':[],
                'other':[]  
            }
    stats_path = os.path.join(folder, 'ACES_private/challenge_set_annotation/stats.txt')
    if args.reset or not os.path.exists(stats_path):
        logger.info('Creating new stats.txt file at {}'.format(stats_path))
        stats = {}
        for p in phenomena_tobe_processed:
            stats[p] = copy.deepcopy(stats_template)
    else:
        logger.info('Path {} already exists. Loading..'.format(stats_path))
        with open(stats_path, "r") as f:
            stats = json.load(f)
        # we want to overwrite the statistics for the new phenomena
        for p in phenomena_tobe_processed:
            stats[p] = copy.deepcopy(stats_template)

    logger.info('Processing running... detokenize: {}'.format(args.detokenize))
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)
    
    # to completely reset the prev annotations
    # annotations = dict()
    stats = {}
    for key in phenomena_tobe_processed:
        stats[key] = copy.deepcopy(stats_template)
    samples = dict()
    for idx, sample in enumerate(dataset):
        if sample['phenomena'] in phenomena_tobe_processed:
            samples[idx] = sample    
    process_phenomena(samples, manual=False, detokenize=args.detokenize)
    
    # using a defaultdict, count the number of examples for each high-levle phenomena
    example_counts = defaultdict(int)
    for sample in annotations:
        example_counts[PHENOMENA_MAPPING[annotations[sample]["phenomena"]]] += 1
    print("Counts:\n", example_counts)

    annotated_dataset_path = os.path.join(folder, 'ACES_private/challenge_set_annotation/annotated.txt')
    with open(annotated_dataset_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)  # encode dict into JSON
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)  # encode dict into JSON
    logger.info("Done writing dict into {} file".format(annotated_dataset_path))