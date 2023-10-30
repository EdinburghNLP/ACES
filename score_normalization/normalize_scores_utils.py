#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv, os, copy, sys
rng = np.random.RandomState(0)

import logging
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from typing import List, Dict, Set

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from aces.cli.evaluate import comp_corr

PHENOMENA = ['addition', 'omission', 'mistranslation', 'untranslated', 'do not translate', 'overtranslation', 
             'undertranslation', 'real-world knowledge', 'wrong language', 'punctuation']

PHENOMENA_MAPPING = {'addition': 'addition',
                     'omission': 'omission',
                     'ambiguous-translation-wrong-discourse-connective-since-causal': 'mistranslation',
                     'ambiguous-translation-wrong-discourse-connective-since-temporal': 'mistranslation',
                     'ambiguous-translation-wrong-discourse-connective-while-contrast': 'mistranslation',
                     'ambiguous-translation-wrong-discourse-connective-while-temporal': 'mistranslation',
                     'ambiguous-translation-wrong-gender-female-anti': 'mistranslation',
                     'ambiguous-translation-wrong-gender-female-pro': 'mistranslation',
                     'ambiguous-translation-wrong-gender-male-anti': 'mistranslation',
                     'ambiguous-translation-wrong-gender-male-pro': 'mistranslation',
                     # 'ambiguous-translation-wrong-sense-frequent': 'mistranslation',
                     # 'ambiguous-translation-wrong-sense-infrequent': 'mistranslation',
                     'anaphoric_group_it-they:deletion': 'mistranslation',
                     'anaphoric_group_it-they:substitution': 'mistranslation',
                     'anaphoric_intra_non-subject_it:deletion': 'mistranslation',
                     'anaphoric_intra_non-subject_it:substitution': 'mistranslation',
                     'anaphoric_intra_subject_it:deletion': 'mistranslation',
                     'anaphoric_intra_subject_it:substitution': 'mistranslation',
                     'anaphoric_intra_they:deletion': 'mistranslation',
                     'anaphoric_intra_they:substitution': 'mistranslation',
                     'anaphoric_singular_they:deletion': 'mistranslation',
                     'anaphoric_singular_they:substitution': 'mistranslation',
                     'coreference-based-on-commonsense': 'mistranslation',
                     'hallucination-date-time': 'mistranslation',
                     'hallucination-named-entity-level-1': 'mistranslation',
                     'hallucination-named-entity-level-2': 'mistranslation',
                     'hallucination-named-entity-level-3': 'mistranslation',
                     'hallucination-number-level-1': 'mistranslation',
                     'hallucination-number-level-2': 'mistranslation',
                     'hallucination-number-level-3': 'mistranslation',
                     'hallucination-real-data-vs-ref-word': 'mistranslation',
                     'hallucination-real-data-vs-synonym': 'mistranslation',
                     'hallucination-unit-conversion-amount-matches-ref': 'mistranslation',
                     'hallucination-unit-conversion-unit-matches-ref': 'mistranslation',
                     'lexical-overlap': 'mistranslation',
                     'modal_verb:deletion': 'mistranslation',
                     'modal_verb:substitution': 'mistranslation',
                     'nonsense': 'mistranslation',
                     'ordering-mismatch': 'mistranslation',
                     'overly-literal-vs-correct-idiom': 'mistranslation',
                     'overly-literal-vs-explanation': 'mistranslation',
                     'overly-literal-vs-ref-word': 'mistranslation',
                     'overly-literal-vs-synonym': 'mistranslation',
                     'pleonastic_it:deletion': 'mistranslation',
                     'pleonastic_it:substitution': 'mistranslation',
                     'xnli-addition-contradiction': 'mistranslation',
                     'xnli-addition-neutral': 'mistranslation',
                     'xnli-omission-contradiction': 'mistranslation',
                     'xnli-omission-neutral': 'mistranslation',
                     'copy-source': 'untranslated',
                     'untranslated-vs-ref-word': 'untranslated',
                     'untranslated-vs-synonym': 'untranslated',
                     'do-not-translate': 'do not translate',
                     'hyponym-replacement': 'overtranslation',
                     'hypernym-replacement': 'undertranslation',
                     'antonym-replacement': 'real-world knowledge',
                     'commonsense-only-ref-ambiguous': 'real-world knowledge',
                     'commonsense-src-and-ref-ambiguous': 'real-world knowledge',
                     'real-world-knowledge-entailment': 'real-world knowledge',
                     'real-world-knowledge-hypernym-vs-distractor': 'real-world knowledge',
                     'real-world-knowledge-hypernym-vs-hyponym': 'real-world knowledge',
                     'real-world-knowledge-synonym-vs-antonym': 'real-world knowledge',
                     'similar-language-high': 'wrong language',
                     'similar-language-low': 'wrong language',
                     'punctuation:deletion_all': 'punctuation',
                     'punctuation:deletion_commas': 'punctuation',
                     #'punctuation:deletion_quotes': 'punctuation',
                     'punctuation:statement-to-question': 'punctuation'
                    }

# NOTE: These ones are not in WMT metrics: 'MATESE', 'MATESE-QE', 'MEE', 'MEE2', 'MEE4'
# The scores are also missing in the ACES 2022 scores

# NOTE: Also these from the ACES-2023 (but they are not listed in the metrics shared task paper anyway):
# 'Calibri-COMET22', 'Calibri-COMET22-QE'
# These don't exist in WMT yet: 'CometKiwi-XL', 'CometKiwi-XXL','GEMBA-MQM','MEE4','MEE4_stsb_xlm',
# 'MetricX-23', 'MetricX-23-QE','MetricX-23-QE-b','MetricX-23-QE-c','MetricX-23-b','MetricX-23-c','Random-sysname',
# and more..

METRIC_MAPPING_BACK = {'YISI-1':'YiSi-1', 'BERTScore':'BERTscore', 'COMETKiwi':'CometKiwi', 'MATESE':'MaTESe', 'COMET-22':'COMET'}
METRIC_NAMES_MAPPING = {
    # For ACES 2022
    'COMET-QE-Baseline':'COMET-QE',
    'YiSi-1':'YISI-1',
    # For ACES 2023
    'BERTscore':'BERTScore',
    'COMET':'COMET-22',
    'CometKiwi':'COMETKiwi',
    'MaTESe':'MATESE',  
    'spBLEU':'f200spBLEU'
}

METRICS_SKIP = ["MATESE", 'MaTESe', "MEE", "MEE4", "MEE2", "REUSE", "MATESE-QE", "MATESE-QE"]

# ----------------------------------------- Loading Data Functions ----------------------------------------------
def read_file(filename: str) -> pd.DataFrame:
    '''
    Read TSV file and return as pandas DataFrame
    '''
    return pd.read_csv(filename, sep='\t', quoting=csv.QUOTE_NONE)

def load_ACES_scores_summary_2022(skip_metrics:List[str] = METRICS_SKIP) -> Dict[str, Dict[str, float]]:
    '''
    Return the ACES scores for each phenomena and metric 
    (no sensitivity: how many times it gives a better score to the correct translation)
    from the paper
    format: {'BLEU': {'addition': 0.748,
                        'omission': 0.435},
                'COMET': {'addition': 0.748,
                        'omission': 0.435},... }
    '''
    table = "BLEU 0.748 0.435 -0.229 0.353 0.600 -0.838 -0.856 -0.768 0.661 0.638 -2.79\n\
    f101spBLEU 0.662 0.590 -0.084 0.660 0.940 -0.738 -0.826 -0.405 0.638 0.639 -0.09\n\
    f200spBLEU 0.664 0.590 -0.082 0.687 0.920 -0.752 -0.794 -0.394 0.658 0.648 0.06\n\
    chrF 0.642 0.784 0.162 0.781 0.960 -0.696 -0.592 -0.294 0.691 0.743 3.71\n\
    BERTScore 0.880 0.750 0.320 0.767 0.960 -0.110 -0.190 0.031 0.563 0.849 10.65\n\
    BLEURT-20 0.437 0.810 0.429 0.748 0.860 0.200 0.014 0.401 0.533 0.649 12.06\n\
    COMET-20 0.437 0.808 0.378 0.748 0.900 0.314 0.112 0.267 0.033 0.706 12.27\n\
    COMET-QE -0.538 0.397 0.378 0.135 0.120 0.622 0.442 0.322 -0.505 0.251 6.61\n\
    YiSi-1 0.770 0.866 0.356 0.730 0.920 -0.062 -0.076 0.110 0.431 0.734 11.53\n\
    COMET-22 0.333 0.806 0.566 0.536 0.900 0.690 0.538 0.574 -0.318 0.539 16.41\n\
    metricx_xl_DA_2019 0.395 0.852 0.545 0.722 0.940 0.692 0.376 0.740 0.521 0.670 17.29\n\
    metricx_xl_MQM_2020 -0.281 0.670 0.523 0.579 -0.740 0.718 0.602 0.705 -0.126 0.445 13.10\n\
    metricx_xxl_DA_2019 0.303 0.832 0.580 0.762 0.920 0.572 0.246 0.691 0.250 0.630 15.35\n\
    metricx_xxl_MQM_2020 -0.099 0.534 0.578 0.651 0.880 0.752 0.552 0.712 -0.321 0.369 13.54\n\
    MS-COMET-22 -0.219 0.686 0.397 0.504 0.700 0.548 0.290 0.230 0.041 0.508 10.03\n\
    UniTE 0.439 0.876 0.501 0.571 0.920 0.496 0.302 0.624 -0.337 0.793 14.93\n\
    UniTE-ref 0.359 0.868 0.535 0.412 0.840 0.640 0.398 0.585 -0.387 0.709 15.52\n\
    COMETKiwi 0.361 0.830 0.631 0.230 0.780 0.738 0.574 0.582 -0.359 0.490 16.95\n\
    Cross-QE 0.163 0.876 0.546 -0.094 0.320 0.726 0.506 0.446 -0.374 0.455 14.43\n\
    HWTSC-Teacher-Sim -0.031 0.495 0.406 -0.269 0.700 0.552 0.456 0.261 -0.021 0.271 10.09\n\
    HWTSC-TLM -0.363 0.345 0.384 0.154 -0.040 0.544 0.474 0.071 -0.168 0.634 7.00\n\
    KG-BERTScore 0.790 0.812 0.489 -0.456 0.760 0.654 0.528 0.487 0.306 0.255 17.49\n\
    MS-COMET-QE-22 -0.177 0.678 0.439 0.388 0.240 0.518 0.386 0.248 -0.197 0.523 9.95\n\
    UniTE-src 0.285 0.930 0.599 -0.615 0.860 0.698 0.540 0.537 -0.417 0.733 15.70\n\
    Average 0.290 0.713 0.389 0.404 0.735 0.312 0.167 0.282 0.075 0.578 10.91".split("\n")
    table2 = [line.split() for line in table]
    keys = [line[0] for line in table2 if line[0] not in skip_metrics]
    phenomena_table = ["addition", "omission", "mistranslation", "untranslated", "do not translate", "overtranslation", "undertranslation", "real-world knowledge", "wrong language", "punctuation", "all"]
    phenomena_table = ["addition", "omission", "mistranslation", "untranslated", "do not translate", "overtranslation", "undertranslation", "real-world knowledge", "wrong language", "punctuation"]
    values = [{phenomena_table[i] : float(line[1:][i]) for i in range(len(line[1:-1]))} for line in table2 if line[0] not in skip_metrics]
    return dict(zip(keys, values))

def load_ACES_scores_summary_2023(skip_metrics:List[str] = METRICS_SKIP) -> Dict[str, Dict[str, float]]:
    '''
    Return the ACES scores for each phenomena and metric 
    (no sensitivity: how many times it gives a better score to the correct translation)
    from the paper
    format: {'BLEU': {'addition': 0.748,
                        'omission': 0.435},
                'COMET': {'addition': 0.748,
                        'omission': 0.435},... }
    '''
    table = "BERTscore 0.884 0.754 0.336 0.761 0.947 -0.173 -0.276 0.032 0.564 0.911 10.02\n\
    BLEU 0.759 0.422 -0.191 0.337 0.711 -0.834 -0.857 -0.767 0.658 0.746 -2.49\n\
    BLEURT-20 0.452 0.807 0.449 0.766 0.868 0.216 0.022 0.388 0.534 0.714 12.36\n\
    chrF 0.669 0.779 0.173 0.748 0.947 -0.699 -0.588 -0.292 0.690 0.850 3.85\n\
    COMET-22 0.325 0.823 0.423 0.724 0.842 0.509 0.260 0.382 0.086 0.631 13.80\n\
    CometKiwi 0.540 0.920 0.616 -0.077 0.632 0.765 0.601 0.576 -0.301 0.883 18.13\n\
    f200spBLEU 0.676 0.579 -0.067 0.663 0.895 -0.746 -0.793 -0.394 0.661 0.752 0.14\n\
    MS-COMET-QE-22 -0.179 0.685 0.453 0.402 0.342 0.528 0.377 0.263 -0.186 0.591 10.20\n\
    Random-sysname -0.115 -0.110 -0.111 -0.053 -0.053 -0.110 -0.146 -0.245 -0.108 -0.020 -3.42\n\
    YiSi-1 0.774 0.863 0.369 0.742 0.921 -0.048 -0.059 0.113 0.434 0.838 11.79\n\
    eBLEU 0.682 0.676 0.211 0.738 0.895 -0.651 -0.679 -0.041 0.767 0.110 3.57\n\
    embed_llama 0.214 0.457 0.016 0.494 0.447 -0.173 -0.489 -0.164 0.151 0.447 1.10\n\
    MaTESe -0.946 -0.815 -0.284 -0.166 -0.079 0.270 -0.038 -0.227 -1.000 -0.246 -10.57\n\
    MetricX-23 -0.018 0.575 0.604 0.472 0.842 0.792 0.588 0.767 -0.478 0.731 14.39\n\
    MetricX-23-b -0.134 0.628 0.597 0.593 0.895 0.773 0.570 0.750 -0.435 0.474 14.02\n\
    MetricX-23-c -0.008 0.802 0.645 0.631 0.842 0.738 0.522 0.783 -0.621 0.588 15.19\n\
    partokengram_F 0.089 0.195 -0.039 0.275 0.132 -0.052 -0.028 0.032 0.507 0.140 1.78\n\
    tokengram_F 0.714 0.752 0.170 0.745 0.947 -0.726 -0.630 -0.272 0.686 0.928 3.60\n\
    XCOMET-Ensemble 0.323 0.788 0.674 0.388 0.816 0.792 0.611 0.708 -0.410 0.825 17.52\n\
    XCOMET-XL 0.184 0.544 0.586 0.228 0.842 0.653 0.464 0.582 -0.345 0.475 13.50\n\
    XCOMET-XXL -0.130 0.415 0.552 0.207 0.658 0.736 0.572 0.508 -0.508 0.625 11.65\n\
    XLsim 0.452 0.634 0.164 0.629 0.842 -0.198 -0.278 -0.042 0.399 0.780 5.78\n\
    cometoid22-wmt21 -0.330 0.664 0.499 -0.064 0.395 0.669 0.562 0.364 -0.450 0.520 10.61\n\
    cometoid22-wmt22 -0.282 0.685 0.499 -0.108 0.368 0.688 0.539 0.340 -0.475 0.520 10.82\n\
    cometoid22-wmt23 -0.259 0.706 0.512 -0.007 0.474 0.748 0.582 0.363 -0.326 0.533 12.00\n\
    CometKiwi-XL 0.255 0.828 0.634 0.262 0.447 0.759 0.557 0.562 -0.366 0.706 16.14\n\
    CometKiwi-XXL 0.371 0.830 0.652 0.404 0.289 0.767 0.555 0.682 -0.525 0.747 16.80\n\
    GEMBA-MQM 0.042 0.300 0.196 0.118 0.184 0.462 0.274 0.268 -0.145 0.105 6.80\n\
    KG-BERTScore 0.540 0.914 0.583 -0.177 0.816 0.771 0.603 0.593 -0.301 0.744 18.06\n\
    MetricX-23-QE 0.042 0.680 0.658 0.384 0.500 0.769 0.607 0.654 -0.695 0.594 14.69\n\
    MetricX-23-QE-b 0.033 0.771 0.665 0.498 0.553 0.759 0.611 0.648 -0.664 0.606 15.29\n\
    MetricX-23-QE-c -0.113 0.666 0.717 0.380 0.447 0.723 0.613 0.754 -0.705 0.491 13.96\n\
    XCOMET-QE-Ensemble 0.282 0.754 0.645 0.200 0.737 0.759 0.584 0.626 -0.508 0.745 16.25\n\
    XLsimQE 0.199 0.396 0.094 -0.703 0.921 0.453 0.357 0.043 0.310 0.745 8.14\n\
    Average 0.206 0.599 0.374 0.336 0.625 0.315 0.181 0.275 -0.091 0.583 9.57".split("\n")
    table2 = [line.split() for line in table]
    keys = [line[0] for line in table2 if line[0] not in skip_metrics]
    phenomena_table = ["addition", "omission", "mistranslation", "untranslated", "do not translate", "overtranslation", "undertranslation", "real-world knowledge", "wrong language", "punctuation", "all"]
    phenomena_table = ["addition", "omission", "mistranslation", "untranslated", "do not translate", "overtranslation", "undertranslation", "real-world knowledge", "wrong language", "punctuation"]
    values = [{phenomena_table[i] : float(line[1:][i]) for i in range(len(line[1:-1]))} for line in table2 if line[0] not in skip_metrics]
    return dict(zip(keys, values))

def load_ACES_scores(ACES_scores_path: str, good_token:str = '.-good', bad_token:str ='.-bad', mapping:Dict[str, str] = METRIC_NAMES_MAPPING, 
                     skip_metrics:List[str] = METRICS_SKIP) -> Dict[str, Dict[str, List[List]]]: 
    '''
    Return the metric scores for each phenomena and metric and samples
    format = {
        'BLEU': {
            'addition': [[1,2,3,…],	# list of scores for the good translations
                        [1,2,3,…]],	# list of scores for the bad translations
            , …, 'all': [[1,2,3,…],	# list of scores for the good translations for all phenomena
                        [1,2,3,…]]	# list of scores for the bad translations for all phenomena
            }
    }
    '''
    if not os.path.exists(ACES_scores_path):
        logger.error('No ACES scores path: %s' %(ACES_scores_path))
        return
    logger.info('Loading ACES the scores...')
    ACES_scores = read_file(ACES_scores_path)
    phenomena = np.unique(ACES_scores['phenomena'])

    ACES_metrics = []
    for key in ACES_scores.keys():
        if good_token in key:
            ACES_metrics.append(key[:-len(good_token)])
        elif bad_token in key:
            ACES_metrics.append(key[:-len(bad_token)])
        if len(ACES_metrics) > 0 and ACES_metrics[-1] in skip_metrics:
            ACES_metrics = ACES_metrics[:-1]
    metrics_names = np.unique(ACES_metrics)

    metric_mapping = mapping.copy()
    if metric_mapping == None:
        metric_mapping = dict(zip(list(metrics_names), list(metrics_names)))
    else:
        for metric in metrics_names:
            if metric not in metric_mapping:
                metric_mapping[metric] = metric

    template = dict(zip(phenomena, np.empty((len(phenomena),2))))
    ACES_metrics = {}
    for metric in metrics_names:
        metric = metric_mapping[metric]
        if metric not in []:
            ACES_metrics[metric] = copy.deepcopy(template)
    for p in phenomena:
        ids = np.where(ACES_scores['phenomena']==p)[0]
        for metric in metrics_names:
            if type(ACES_metrics[metric_mapping[metric]][p][0]) != list:
                ACES_metrics[metric_mapping[metric]][p] = [list(ACES_scores[metric+good_token][ids]), list(ACES_scores[metric+bad_token][ids])]
            else:     
                ACES_metrics[metric_mapping[metric]][p][0].extend(list(ACES_scores[metric+good_token][ids]))
                ACES_metrics[metric_mapping[metric]][p][1].extend(list(ACES_scores[metric+bad_token][ids])) 
    """
    for metric in metrics_names:
        if "all" not in ACES_metrics[metric_mapping[metric]]:
            ACES_metrics[metric_mapping[metric]]["all"] = [list(ACES_scores[metric+good_token]), list(ACES_scores[metric+bad_token])]
        else:
            ACES_metrics[metric_mapping[metric]]["all"][0].extend(list(ACES_scores[metric+good_token]))
            ACES_metrics[metric_mapping[metric]]["all"][1].extend(list(ACES_scores[metric+bad_token])) 
    """
    return ACES_metrics

def map_to_higher(ACES_scores: Dict[str, Dict[str, List[List]]], mapping: Dict=PHENOMENA_MAPPING) -> Dict[str, Dict[str, List[List]]]:
    '''
    Return the ACES scores dict where the phenomena are mapped to higher level phenomena
    inputs:
        ACES_metrics: output of load_ACES_scores function
        mapping: constant PHENOMENA_MAPPING. Different phenomena mapping dict can be used too
    '''
    ACES_scores = copy.deepcopy(ACES_scores)
    phenomena = list(ACES_scores['BLEU'].keys())
    higher_level_phenomena = list(set(mapping.values()))
    template = dict(zip(higher_level_phenomena, np.zeros((len(higher_level_phenomena)))))
    res = {}

    for metric in ACES_scores:
        res[metric] = copy.deepcopy(template)
        for p in mapping.keys():
            if p in phenomena:
                if type(res[metric][mapping[p]]) != list:
                    res[metric][mapping[p]] = ACES_scores[metric][p]
                else:
                    res[metric][mapping[p]][0].extend(ACES_scores[metric][p][0])
                    res[metric][mapping[p]][1].extend(ACES_scores[metric][p][1])
    return res

def read_wmt_file(filename: str) -> pd.DataFrame:
    '''
    Read txt file and return a list of all the scores in it.
    '''
    with open(filename, 'r') as f:
        lines  = f.readlines()
    scores = [float(line.split('\t')[1]) for line in lines]
    return scores

def read_wmt_file_23(filename: str) -> pd.DataFrame:
    '''
    Read txt file and return a list of all the scores in it.
    '''
    with open(filename, 'r') as f:
        lines  = f.readlines()
    scores = [float(line.split('\t')[-1]) for line in lines]
    return scores

def load_WMT_scores(WMT_scores_path: str, metrics_names: Set[str]) -> Dict[str, list]:
    '''
    Load the scores in wmt-metrics-eval-v2/wmt22/metric-scores/cs-en/
    format = {
        'BLEU': [1,2,3,…],	    # list of BLEU scores evaluated on WMT
        'COMET': [1,2,3,...]    # we don't have separate scores for phenomena or good or bad translations now
    }
    '''
    WMT_metrics = dict()
    langpairs = os.listdir(WMT_scores_path)
    for lang in langpairs:
        print(lang)
        #Locate langpair dir, in there locate BERT-ref-seg.score files
        lang_dir = os.path.join(WMT_scores_path, lang)
        files = os.listdir(lang_dir)
        for metric in tqdm(metrics_names):
            # if the metric name does not exist in the WMT scores list it just skips
            for file in files:
                if file.startswith(metric) and file.endswith(".seg.score"):
                    scores_tmp = read_wmt_file(os.path.join(lang_dir, file))
                    if metric not in WMT_metrics:
                        WMT_metrics[metric] = scores_tmp
                    else:
                        WMT_metrics[metric].extend(scores_tmp)
                elif metric in METRIC_NAMES_MAPPING and file.startswith(METRIC_NAMES_MAPPING[metric]) and file.endswith(".seg.score"):
                    metric = METRIC_NAMES_MAPPING[metric]
                    scores_tmp = read_wmt_file(os.path.join(lang_dir, file))
                    if metric not in WMT_metrics:
                        WMT_metrics[metric] = scores_tmp
                    else:
                        WMT_metrics[metric].extend(scores_tmp)
                elif metric in METRIC_MAPPING_BACK and file.startswith(METRIC_MAPPING_BACK[metric]) and file.endswith(".seg.score"):
                    scores_tmp = read_wmt_file(os.path.join(lang_dir, file))
                    if metric not in WMT_metrics:
                        WMT_metrics[metric] = scores_tmp
                    else:
                        WMT_metrics[metric].extend(scores_tmp)
    return WMT_metrics

def load_WMT_scores_23(WMT_scores_path: str, metrics_names: Set[str]) -> Dict[str, list]:
    '''
    Load the scores in wmt-metrics-eval-v2/wmt22/metric-scores/cs-en/
    format = {
        'BLEU': [1,2,3,…],	    # list of BLEU scores evaluated on WMT
        'COMET': [1,2,3,...]    # we don't have separate scores for phenomena or good or bad translations now
    }
    '''
    WMT_metrics = dict()
    files = os.listdir(WMT_scores_path)
    if metrics_names == None:
        metrics_names = set([file.split('.', 1)[0] for file in files])
    for file in files:
        metric = file.split('.', 1)[0]
        if metric in metrics_names and file.endswith(".seg.score"):
            scores_tmp = read_wmt_file_23(os.path.join(WMT_scores_path, file))
            if metric not in WMT_metrics:
                WMT_metrics[metric] = scores_tmp
            else:
                WMT_metrics[metric].extend(scores_tmp)
        elif metric in METRIC_NAMES_MAPPING and file.endswith(".seg.score"):
            metric = METRIC_NAMES_MAPPING[metric]
            scores_tmp = read_wmt_file_23(os.path.join(WMT_scores_path, file))
            if metric not in WMT_metrics:
                WMT_metrics[metric] = scores_tmp
            else:
                WMT_metrics[metric].extend(scores_tmp)
        elif metric in METRIC_MAPPING_BACK and file.endswith(".seg.score"):
            scores_tmp = read_wmt_file_23(os.path.join(WMT_scores_path, file))
            if metric not in WMT_metrics:
                WMT_metrics[metric] = scores_tmp
            else:
                WMT_metrics[metric].extend(scores_tmp)
    return WMT_metrics
                    
# --------------------------------------- Normalization Functions ----------------------------------------------------
def iqr(dist: list) -> float:
    '''
    Calculate Interquartile range of given distribution
    '''
    return np.subtract(*np.percentile(dist, [75, 25]))

def scale_distribution(ACES_scores: list, WMT_scores: list) -> list:
    '''
    Normalize the subset of ACES scores using the larger distribution, WMT scores
    '''
    return (ACES_scores - np.mean(WMT_scores)) / iqr(WMT_scores)
    
def calculate_sensitivities(ACES_scores: Dict[str, Dict[str, List[List]]], WMT_scores: Dict[str, list], mapping: dict=None, verbal=False) -> Dict[str, Dict[str, Dict[str, int]]]:
    '''
    There are 3 ways we can measure the sensitivity of a metric:
        Mean of good-bad : Probably the best one
        Standart Deviation of good-bad : not that useful
        Mean of abs(good-bad) : not good when the metric scores incorrect-translation more than good-translation
    And 
    This function calculates these stats for each phenomenon in ACES, or mapped ACES if mapping is not None.
    
    Also note: This doesn't really save the normalized score for every single sample (would take too much storage), 
    but those can be individually normalized with scale_distribution function.
    
    format = {
        'addition': {
            'BLEU': {"Mean of good-bad": 15, 
                    "Standart Deviation of good-bad": 3, 
                    "Mean of abs(good-bad)": 4
                    },
            'COMET': ...
        }
    }
    '''
    ACES_scores = copy.deepcopy(ACES_scores)

    if verbal:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
        
    if mapping:
        ACES_scores = map_to_higher(ACES_scores, mapping=mapping)
    
    metrics_names = list(set(ACES_scores.keys()).intersection(set(WMT_scores.keys())))
    phenomena = list(ACES_scores[metrics_names[0]].keys())
    
    stats = {}
    for p in phenomena:
        stats[p] = {}
        logger.info('\n-------------------------{}-----------------------------'.format(p))
        logger.info('Metric\t\tMean of good-bad\t\tStandart Deviation of good-bad\t\tMean of abs(good-bad)')
        for metric in metrics_names:
            good_scaled = scale_distribution(ACES_scores[metric][p][0], WMT_scores[metric])
            bad_scaled = scale_distribution(ACES_scores[metric][p][1], WMT_scores[metric])
            diff_raw_scaled = good_scaled - bad_scaled
            diff_scaled = np.abs(diff_raw_scaled)
            mean_1 = np.mean(diff_raw_scaled)
            std_1 = np.std(diff_raw_scaled)
            mean_2 = np.mean(diff_scaled)
            logger.info('{}\t\t{}\t\t{}\t\t{}'.format(metric, mean_1, std_1, mean_2))
            stats[p][metric] = {"Mean of good-bad":mean_1, "Standart Deviation of good-bad":std_1, "Mean of abs(good-bad)":mean_2, "Mean of good":np.mean(good_scaled), "Mean of bad":np.mean(bad_scaled)}
    means = {metric:{p: stats[p][metric]["Mean of good-bad"] for p in phenomena} for metric in metrics_names}
    abs_means = {metric:{p: stats[p][metric]["Mean of abs(good-bad)"] for p in phenomena} for metric in metrics_names}
    normal_std = {metric:{p: stats[p][metric]["Standart Deviation of good-bad"] for p in phenomena} for metric in metrics_names}
    means_good = {metric:{p: stats[p][metric]["Mean of good"] for p in phenomena} for metric in metrics_names}
    means_bad = {metric:{p: stats[p][metric]["Mean of bad"] for p in phenomena} for metric in metrics_names}
    return means, abs_means, normal_std, phenomena, means_good, means_bad

def calculate_sensitivities_self_scaled(ACES_scores: Dict[str, Dict[str, List[List]]], mapping: dict=None, verbal=False) -> Dict[str, Dict[str, Dict[str, int]]]:
    ACES_scores = copy.deepcopy(ACES_scores)
    
    if verbal:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
        
    if mapping:
        ACES_scores = map_to_higher(ACES_scores, mapping=mapping)
    
    metrics_names = list(set(ACES_scores.keys()))
    phenomena = list(ACES_scores[metrics_names[0]].keys())
    
    stats = {}
    for p in phenomena:
        stats[p] = {}
        logger.info('\n-------------------------{}-----------------------------'.format(p))
        logger.info('Metric\t\tMean of good-bad\t\tStandart Deviation of good-bad\t\tMean of abs(good-bad)')
        for metric in metrics_names:
            good = np.array(ACES_scores[metric][p][0])
            bad = np.array(ACES_scores[metric][p][1])
            sensitivity = np.sum(good-bad) / np.sum(np.abs(good)+np.abs(bad))
            stats[p][metric] = {"Sensitivity unscaled":sensitivity}
    sensitivities = {metric:{p: stats[p][metric]["Sensitivity unscaled"] for p in phenomena} for metric in metrics_names}
    logger.setLevel(logging.INFO)
    return sensitivities, phenomena

def calculate_tau_correlations(ACES_scores: Dict[str, Dict[str, List[List]]], mapping:dict=None, phenomena:List[str]=None) -> Dict[str, Dict[str, Dict[str, int]]]:
    ACES_scores = copy.deepcopy(ACES_scores)
    
    if mapping:
        ACES_scores = map_to_higher(ACES_scores, mapping=mapping)
    
    metrics_names = list(set(ACES_scores.keys()))
    if phenomena == None:                  
        phenomena = list(ACES_scores[metrics_names[0]].keys())
            
    stats = {}
    for p in phenomena:
        stats[p] = {}
        for metric in metrics_names:
            good = pd.Series(ACES_scores[metric][p][0])
            bad = pd.Series(ACES_scores[metric][p][1])
            tau = comp_corr(good, bad)
            stats[p][metric] = {"Tau":tau}
    sensitivities = {metric:{p: stats[p][metric]["Tau"] for p in phenomena} for metric in metrics_names}
    logger.setLevel(logging.INFO)
    return sensitivities, phenomena

# -----------------------------------------------Plotting Functions ---------------------------------------------------
def mean_var(dist: list):
    '''
    return a str with the mean and variance of the given dist to be used in plots
    '''
    return "_"+str(round(np.mean(dist),2)) + "_" + str(round(np.std(dist),2))

def plot(dists: List[list], names: List[str]):    
    '''
    given a list of distributions and a list of their names, plot all the distributions in one plot
    plot([COMET_all_scaled_wmt, COMET_all_wmt, COMET_all, COMET_all_scaled], ['COMET_all_scaled_wmt', 'COMET_all_wmt', 'COMET_all', 'COMET_all_scaled'])
    '''
    fig = go.Figure()
    for i in range(len(dists)):
        d = np.array(dists[i])
        if len(d.shape) > 1:
            d = np.squeeze(d)
        fig.add_trace(go.Histogram(x=d, name=names[i] + mean_var(dists[i])))

    fig.update_layout(barmode='overlay', xaxis_title_text='Score', yaxis_title_text='Count')
    fig.update_traces(opacity=0.60)
    fig.show()
    
def grouped_line_plot(groups: List[Dict[str,list]], metrics_names: List[str], group_labels: List[str], phenomena: List[str]):
    '''
    Inputs: 
        1. means and tau scores
        format = {
            metric1: [score for phenomenon 1, score for phenomenon 2, ..]
        }
        2. A list of the labels for: 
            the groups (mean (good-bad), tau, ...)
            metrics
            phenomena (the order is important because in means and tau scores the scores are ordered acc. to phenomena)
    '''
    assert len(groups) > 0 and len(groups) == len(group_labels) and len(metrics_names) > 0
    fig = go.Figure()
    colors = [['lightsteelblue',  'aqua', 'aquamarine', 'darkturquoise'],
        ['chocolate', 'coral', 'crimson', 'orange']]
    for i,group in enumerate(groups):
        for j,metric in enumerate(metrics_names):
            fig.add_trace(go.Scatter(x=phenomena, y=group[metric],mode='lines',name=group_labels[i]+" - "+metric,
                          line=dict(color=colors[i][j])))
    fig.update_layout(
        title=" ".join(group_labels)
        )
    fig.show()

# ------------------------------------ LATEX FUNCTIONS --------------------------------
import decimal
import re

METRICS_GROUPING_2022= {"baseline": ["BLEU", "f101spBLEU", "f200spBLEU", "chrF", "BERTScore", "BLEURT-20",
                                     "COMET-20", "COMET-QE", "YISI-1"],
                        "reference-based": ["COMET-22", 'metricx_xl_DA_2019', 'metricx_xl_MQM_2020', 'metricx_xxl_DA_2019', 'metricx_xxl_MQM_2020',
                                            'MS-COMET-22', "UniTE", "UniTE-ref"],
                        "reference-free": ["COMETKiwi", "Cross-QE", 'HWTSC-Teacher-Sim', 'HWTSC-TLM',
                                           'KG-BERTScore', "MS-COMET-QE-22", "UniTE-src"]
                    }
METRICS_GROUPING_2023 = {
    "baseline": ["BERTScore", "BLEU", "BLEURT-20", "chrF", "COMET-22", "COMETKiwi", 
                 "f200spBLEU", "MS-COMET-QE-22", "Random-sysname", "YiSi-1"],
    "reference-based": ["eBLEU", "embed_llama", "MaTESe", "MetricX-23", "MetricX-23-b", 
                        "MetricX-23-c", "partokengram_F", "tokengram_F", "XCOMET-Ensemble",
                        "XCOMET-XL", "XCOMET-XXL", "XLsim"],
    "reference-free": ["cometoid22-wmt21", "cometoid22-wmt22", "cometoid22-wmt23", "CometKiwi-XL", 
                       "CometKiwi-XXL", "GEMBA-MQM", "KG-BERTScore", "MetricX-23-QE", "MetricX-23-QE-b", 
                       "MetricX-23-QE-c", "XCOMET-QE-Ensemble", "XLsimQE"]                           
}

PHENOMENA_HEADER_1 = dict(zip(PHENOMENA, ['\\hyperref[sec:addition-omission]{\\textbf{addition}}', '\hyperref[sec:addition-omission]{\\textbf{omission}}', '\hyperref[sec:source-disambig]{\\textbf{mistranslation}}', 
                      '\hyperref[sec:untranslated]{\\textbf{untranslated}}', '\hyperref[sec:do-not-translate]{\\textbf{do not}}', 
                      '\hyperref[sec:overtranslation_undertranslation]{\\textbf{overtranslation}}', '\hyperref[sec:overtranslation_undertranslation]{\\textbf{undertranslation}}',
                        '\hyperref[sec:real-world-knowledge]{\\textbf{real-world}}', '\hyperref[sec:wrong_language]{\\textbf{wrong}}', '\hyperref[sec:punctuation]{\\textbf{punctuation}}',
                          '\\textbf{ACES-}\\']))
PHENOMENA_HEADER_2 = dict(zip(PHENOMENA, ['', '',  '',  '', '\\hyperref[sec:do-not-translate]{\\textbf{translate}}', '',  '', '\hyperref[sec:real-world-knowledge]{\\textbf{knowledge}}',
                            '\hyperref[sec:wrong_language]{\\textbf{language}}', '', '\\textbf{Score}']))
NUM_SAMPLES = dict(zip(PHENOMENA, [999, 999, 24457, 1300, 100, 1000, 1000, 2948, 2000, 1673]))
SUMMARY_AVERAGES_2022 = dict(zip(PHENOMENA, [0.290, 0.713, 0.389, 0.404, 0.673, 0.312, 0.167, 0.282, 0.075, 0.578]))
SENS_AVERAGES_2022 = dict(zip(PHENOMENA, [0.061, 0.310, 0.250, 0.788, 0.319, 0.211, 0.080, 0.145, -0.086, 0.235]))

# Overwriting from aces/cli/score-py
def comp_aces_score(overview_results: List[float]) -> float:
    '''
    Compute weighted summary score based on top-level correlation scores.
    '''
    for p in PHENOMENA:
        if p not in overview_results:
            overview_results[p] = [0]
    aces_score = 5 * np.mean(overview_results['addition']) + \
                 5 * np.mean(overview_results['omission']) + \
                 5 * np.mean(overview_results['mistranslation']) +\
                 1 * np.mean(overview_results['untranslated']) + \
                 1 * np.mean(overview_results['do not translate']) + \
                 5 * np.mean(overview_results['overtranslation']) + \
                 5 * np.mean(overview_results['undertranslation']) + \
                 1 * np.mean(overview_results['real-world knowledge']) + \
                 1 * np.mean(overview_results['wrong language']) + \
                 0.1 * np.mean(overview_results['punctuation'])
    return aces_score

def format_number(number:float, max_phenomena:bool = False, dec:str = '0.000', color='\colorbox[HTML]{B2EAB1}') -> str:
    if number == -np.inf:
        return '----'
    if number == 0.0:
        return '---'
    if number < 0:
        out = '-' + str(decimal.Decimal(-number).quantize(decimal.Decimal(dec)))
    else:
        out = '\phantom{-}' + str(decimal.Decimal(number).quantize(decimal.Decimal(dec)))
    if max_phenomena:
        return color + '{\\textbf{' + out + '}}'
    else:
        return out
    
def format_metric(metric:str) -> str:
    return re.sub(r'_', '\\_', metric)

def map_to_color(num:float, max:float, min:float) -> str:
    # colors = ["red5", "red3", "red2", "red1", "white", "white", "green1", "green2", "green3", "green5", "green5"]
    colors = ["red5","red3", "red2", "red1", "white", "white", "green1", "green2", "green3", "green5", "green5"]
    # return '\colorbox{' + colors[int((num-min)/(max-min)*10)] + '}'
    # print(num, max, min, int(num/(2*max)*10+5))
    if np.abs(max) > np.abs(min):
        return '\colorbox{' + colors[np.maximum(0, int(num/(2*np.abs(max))*10+5))] + '}'
    else:
        return '\colorbox{' + colors[np.maximum(0, int(num/(2*np.abs(min))*10+5))] + '}'

def find_max_on_col(scores:Dict[str, Dict[str, Dict[str, int]]], metrics_names:List[str], phenomena:List[str]=PHENOMENA, k_highest:int=1) -> Dict[str,str]:
    max_metrics = {str(k): [[] for metric in metrics_names] for k in range(k_highest)}
    avgs = []
    for i,p in enumerate(phenomena):
        col = []
        for metric in metrics_names:
            if metric not in scores and metric in METRIC_NAMES_MAPPING:
                metric = METRIC_NAMES_MAPPING[metric]
            elif metric not in scores and metric in METRIC_MAPPING_BACK:
                metric = METRIC_MAPPING_BACK[metric]
            if metric not in scores:
                col.append(-np.inf)
            else:
                col.append(scores[metric][p])
        for k in range(k_highest):
            max_ids = np.where(col == np.partition(col, -k-1)[-k-1])[0]
            for max_id in max_ids:
                max_metrics[str(k)][max_id].append(i)
        col = np.array(col)
        avgs.append(np.average(col[col > -np.inf]))
    return max_metrics, avgs

def make_header(scores:Dict[str, Dict[str, Dict[str, int]]], phenomena:List[str]=PHENOMENA, p_header_1:dict=PHENOMENA_HEADER_1, p_header_2:dict=PHENOMENA_HEADER_2, num_samples:dict=None, phenomena_mapping:Dict[str,str]=None, ACES_column:bool=True) -> str:
    if phenomena_mapping == None:
        phenomena_mapping = PHENOMENA_MAPPING
    if num_samples == None:
        num_samples = {p:0 for p in phenomena}
        for p,target in phenomena_mapping.items():
            num_samples[target] += len(scores["BLEU"][p][0])
    res = "\\begin{table*}[ht] \n \small \n \setlength{\\tabcolsep}{3.75pt} \n \centering \n \\begin{tabular}{@{}lccccccccccc@{}} \n \\\\\\toprule \n"
    for p in phenomena:
        res += " & "
        res += p_header_1[p]
    if ACES_column:
        res += """ & \\textbf{ACES-}"""
    res += " \\\\\n"
    for p in phenomena:
        res += " & "
        res += p_header_2[p]
    if ACES_column:
        res += """ & \\textbf{Score}"""
    res += "\\\\\n\midrule\n\\textit{\\textbf{Examples}} "
    for p in phenomena:
        res += " & "
        res += '\\textit{' + str(num_samples[p]) + '}'
    res += """\\\\ \n \midrule"""
    return res

def make_footer(averages:List, phenomena:List[str]=PHENOMENA) -> str:
    res = "\midrule\nAverage (all metrics)\t"
    for p in phenomena:
        if p in phenomena:
            res += " & "
            res += format_number(averages[p])
    res += """\\\\ \n \\bottomrule"""
    return res

def generate_summary_table(scores:Dict[str, Dict[str, Dict[str, int]]], metrics_groups:Dict[str,list] = METRICS_GROUPING_2022, phenomena:List[str]=PHENOMENA, ACES_column:bool=True, global_colors:bool=True, k_highest:int=1, colors:List[str]=None) -> str:
    """
    if k_highest % 2 == 1:
        colors = COLORS[len(COLORS)//2-k_highest//2:len(COLORS)//2+k_highest//2+1]
    else:
        colors = COLORS[len(COLORS)//2-k_highest//2:len(COLORS)//2] + COLORS[len(COLORS)//2+1:len(COLORS)//2+k_highest//2+1]
    """
    if global_colors:
        k_highest = 1
    out = ''
    metrics_names = []
    for group in metrics_groups.values():
        for metric in group:
            if metric not in scores and metric in METRIC_NAMES_MAPPING:
                metrics_names.append(METRIC_NAMES_MAPPING[metric])
            elif metric not in scores and metric in METRIC_MAPPING_BACK:
                metrics_names.append(METRIC_MAPPING_BACK[metric])
            else:
                metrics_names.append(metric)
    # print(metrics_names)
    max_in_columns, avgs = find_max_on_col(scores, metrics_names=metrics_names, phenomena=phenomena, k_highest=k_highest)
    if ACES_column:
        aces_scores_col = []
        for metric in metrics_names:
            # print(metric)
            if metric not in scores and metric in METRIC_NAMES_MAPPING:
                metric = METRIC_NAMES_MAPPING[metric]
            elif metric not in scores and metric in METRIC_MAPPING_BACK:
                metric = METRIC_MAPPING_BACK[metric]
            row = {}
            # print(metric)
            for p_id, p in enumerate(phenomena):
                if metric not in scores:
                    row[p] = 0.0
                else:
                    row[p] = scores[metric][p]
            aces_scores_col.append(comp_aces_score(row))
        # print(aces_scores_col)
        aces_scores_col_colors = {m_id:"" for m_id in range(len(metrics_names))}
    
    if global_colors:
        maximum = np.max([np.max(list(p.values())) for metric,p in scores.items() if metric in metrics_names])
        minimum = np.min([np.min(list(p.values())) for metric,p in scores.items() if metric in metrics_names])
        # print(minimum, maximum, metrics_names)
    if ACES_column:
        if global_colors:
            for i in range(len(aces_scores_col)):
                aces_scores_col_colors[i] = map_to_color(aces_scores_col[i], np.max(aces_scores_col), np.min(aces_scores_col))
        elif k_highest == 1:
            max_aces_ids = np.where(list(aces_scores_col) == np.max(aces_scores_col))[0]
            for i in max_aces_ids:
                aces_scores_col_colors[i] = '\colorbox[HTML]{B2EAB1}'
        else:
            for k in range(k_highest):
                max_aces_ids = np.where(aces_scores_col == np.partition(aces_scores_col, -k-1)[-k-1])[0]
                for i in max_aces_ids:
                    aces_scores_col_colors[i] = colors[k]
                   
    for group, metrics in metrics_groups.items():
        for m_id, metric in enumerate(metrics):
            if metric not in scores and metric in METRIC_NAMES_MAPPING:
                metric = METRIC_NAMES_MAPPING[metric]
            elif metric not in scores and metric in METRIC_MAPPING_BACK:
                metric = METRIC_MAPPING_BACK[metric]
            out += format_metric(metric) + '\t\t\t\t\t'
            for p_id, p in enumerate(phenomena):
                if metric not in scores:
                    out += '&\t ---- \t' 
                else:
                    if global_colors:
                        out += '&\t' + format_number(scores[metric][p], max_phenomena=True, color=map_to_color(scores[metric][p], maximum, minimum)) + '\t'   
                    elif k_highest == 1:
                        max_ids = max_in_columns['0'][metrics_names.index(metric)]
                        out += '&\t' + format_number(scores[metric][p], max_phenomena=p_id in max_ids) + '\t'   
                    else:
                        for k in range(k_highest):
                            max_ids = max_in_columns[str(k)][metrics_names.index(metric)]
                            if p_id in max_ids:
                                color=colors[k]
                                break
                        out += '&\t' + format_number(scores[metric][p], max_phenomena=True, color=color) + '\t'   
            if ACES_column:
                tmp_color =  aces_scores_col_colors[metrics_names.index(metric)]    
                out += '&\t' + format_number(aces_scores_col[metrics_names.index(metric)], dec='0.00', max_phenomena=tmp_color!="", color=tmp_color)
            out += '\t \\\\ \n'
        out += '\midrule \n'
    out += 'Average\t\t\t\t\t'
    for p_id, p in enumerate(phenomena):
        out += '&\t' + format_number(avgs[p_id], max_phenomena=True, color=map_to_color(avgs[p_id], max=np.max(avgs), min=np.min(avgs))) + '\t'
    out += '\\\\'
    return out

def generate_summary_table_double(scores1:Dict[str, Dict[str, Dict[str, int]]], scores2:Dict[str, Dict[str, Dict[str, int]]], 
                                  metrics_groups:Dict[str,list] = METRICS_GROUPING_2022, phenomena:List[str]=PHENOMENA, ACES_column:bool=True, global_colors:bool=True, k_highest:int=1, colors:List[str]=None) -> str:
    """
    if k_highest % 2 == 1:
        colors = COLORS[len(COLORS)//2-k_highest//2:len(COLORS)//2+k_highest//2+1]
    else:
        colors = COLORS[len(COLORS)//2-k_highest//2:len(COLORS)//2] + COLORS[len(COLORS)//2+1:len(COLORS)//2+k_highest//2+1]
    """
    if global_colors:
        k_highest = 1
    out = ''
    metrics_names = []
    for group in metrics_groups.values():
        for metric in group:
            if metric not in scores1 and metric in METRIC_NAMES_MAPPING:
                metrics_names.append(METRIC_NAMES_MAPPING[metric])
            elif metric not in scores1 and metric in METRIC_MAPPING_BACK:
                metrics_names.append(METRIC_MAPPING_BACK[metric])
            else:
                metrics_names.append(metric)
    # print(metrics_names)
    max_in_columns1, avgs1 = find_max_on_col(scores1, metrics_names=metrics_names, phenomena=phenomena, k_highest=k_highest)
    max_in_columns2, avgs2 = find_max_on_col(scores2, metrics_names=metrics_names, phenomena=phenomena, k_highest=k_highest)
    
    if global_colors:
        maximum1 = np.max([np.max([v for p,v in value.items() if p in phenomena]) for metric,value in scores1.items() if metric in metrics_names])
        minimum1 = np.min([np.min([v for p,v in value.items() if p in phenomena]) for metric,value in scores1.items() if metric in metrics_names])
        maximum2 = np.max([np.max([v for p,v in value.items() if p in phenomena]) for metric,value in scores2.items() if metric in metrics_names])
        minimum2 = np.min([np.min([v for p,v in value.items() if p in phenomena]) for metric,value in scores2.items() if metric in metrics_names])
   
    for group, metrics in metrics_groups.items():
        for m_id, metric in enumerate(metrics):
            if metric not in scores1 and metric in METRIC_NAMES_MAPPING:
                metric = METRIC_NAMES_MAPPING[metric]
            elif metric not in scores1 and metric in METRIC_MAPPING_BACK:
                metric = METRIC_MAPPING_BACK[metric]
            out += format_metric(metric) + '\t\t\t\t\t'
            for p_id, p in enumerate(phenomena):
                if metric not in scores1:
                    out += '&\t ---- \t' 
                else:
                    if global_colors:
                        out += '&\t' + format_number(scores1[metric][p], max_phenomena=True, color=map_to_color(scores1[metric][p], maximum1, minimum1)) + '\t'   
                    elif k_highest == 1:
                        max_ids = max_in_columns1['0'][metrics_names.index(metric)]
                        out += '&\t' + format_number(scores1[metric][p], max_phenomena=p_id in max_ids) + '\t'   
                    else:
                        for k in range(k_highest):
                            max_ids = max_in_columns1[str(k)][metrics_names.index(metric)]
                            if p_id in max_ids:
                                color=colors[k]
                                break
                        out += '&\t' + format_number(scores1[metric][p], max_phenomena=True, color=color) + '\t' 
            for p_id, p in enumerate(phenomena):
                if metric not in scores2:
                    out += '&\t ---- \t' 
                else:
                    if global_colors:
                        out += '&\t' + format_number(scores2[metric][p], max_phenomena=True, color=map_to_color(scores2[metric][p], maximum2, minimum2)) + '\t'   
                    elif k_highest == 1:
                        max_ids = max_in_columns2['0'][metrics_names.index(metric)]
                        out += '&\t' + format_number(scores2[metric][p], max_phenomena=p_id in max_ids) + '\t'   
                    else:
                        for k in range(k_highest):
                            max_ids = max_in_columns2[str(k)][metrics_names.index(metric)]
                            if p_id in max_ids:
                                color=colors[k]
                                break
                        out += '&\t' + format_number(scores2[metric][p], max_phenomena=True, color=color) + '\t'    
            out += '\t \\\\ \n'
        out += '\midrule \n'
    out += 'Average\t\t\t\t\t'
    for p_id, p in enumerate(phenomena):
        out += '&\t' + format_number(avgs1[p_id], max_phenomena=True, color=map_to_color(avgs1[p_id], max=np.max(avgs1), min=np.min(avgs1))) + '\t'
        out += '&\t' + format_number(avgs2[p_id], max_phenomena=True, color=map_to_color(avgs2[p_id], max=np.max(avgs2), min=np.min(avgs2))) + '\t'
    out += '\\\\'
    return out