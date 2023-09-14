#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import csv, os, copy, glob
from sklearn.preprocessing import RobustScaler, QuantileTransformer
rng = np.random.RandomState(0)

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

import plotly.graph_objects as go
from tqdm import tqdm
from typing import List, Dict, Set

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
                     'ambiguous-translation-wrong-sense-frequent': 'mistranslation',
                     'ambiguous-translation-wrong-sense-infrequent': 'mistranslation',
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
                     'punctuation:deletion_quotes': 'punctuation',
                     'punctuation:statement-to-question': 'punctuation'
                    }

# ----------------------------------------- Loading Data Functions ----------------------------------------------
def read_file(filename: str) -> pd.DataFrame:
    '''
    Read TSV file and return as pandas DataFrame
    '''
    return pd.read_csv(filename, sep='\t', quoting=csv.QUOTE_NONE)

def load_ACES_scores_summary_2022() -> Dict[str, Dict[str, float]]:
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
    keys = [line[0] for line in table2]
    phenomena_table = ["addition", "omission", "mistranslation", "untranslated", "do not translate", "overtranslation", "undertranslation", "real-world knowledge", "wrong language", "punctuation", "all"]
    values = [{phenomena_table[i] : float(line[1:][i]) for i in range(len(line[1:]))} for line in table2]
    return dict(zip(keys, values))

def load_ACES_scores(ACES_scores_path: str) -> Dict[str, Dict[str, List[List]]]: 
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
        exit()
    logger.info('Loading ACES the scores...')
    ACES_scores = read_file(ACES_scores_path)
    phenomena = np.unique(ACES_scores['phenomena'])

    ACES_metrics = []
    for key in ACES_scores.keys():
        if ".-good" in key:
            ACES_metrics.append(key[:-6])
        elif ".-bad" in key:
            ACES_metrics.append(key[:-5])
    metrics_names = np.unique(ACES_metrics)

    template = dict(zip(phenomena, np.empty((len(phenomena),2))))
    ACES_metrics = {}
    for metric in metrics_names:
        ACES_metrics[metric] = copy.deepcopy(template)

    for p in phenomena:
        ids = np.where(ACES_scores['phenomena']==p)[0]
        for metric in metrics_names:
            ACES_metrics[metric][p] = [list(ACES_scores[metric+'.-good'][ids]), list(ACES_scores[metric+'.-bad'][ids])]
    for metric in metrics_names:
        ACES_metrics[metric]["all"] = [list(ACES_scores[metric+'.-good']), list(ACES_scores[metric+'.-bad'])]
    return ACES_metrics

def map_to_higher(ACES_scores: Dict[str, Dict[str, List[List]]], mapping: Dict=PHENOMENA_MAPPING) -> Dict[str, Dict[str, List[List]]]:
    '''
    Return the ACES scores dict where the phenomena are mapped to higher level phenomena
    inputs:
        ACES_metrics: output of load_ACES_scores function
        mapping: constant PHENOMENA_MAPPING. Different phenomena mapping dict can be used too
    '''
    higher_level_phenomena = list(set(mapping.values()))
    template = dict(zip(higher_level_phenomena, np.zeros((len(higher_level_phenomena)))))
    res = {}

    for metric in ACES_scores:
        res[metric] = copy.deepcopy(template)
        for p in mapping.keys():
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
            stats[p][metric] = {"Mean of good-bad":mean_1, "Standart Deviation of good-bad":std_1, "Mean of abs(good-bad)":mean_2}
    means = {metric:{p: stats[p][metric]["Mean of good-bad"] for p in phenomena} for metric in metrics_names}
    abs_means = {metric:{p: stats[p][metric]["Mean of abs(good-bad)"] for p in phenomena} for metric in metrics_names}
    normal_std = {metric:{p: stats[p][metric]["Standart Deviation of good-bad"] for p in phenomena} for metric in metrics_names}
    return means, abs_means, normal_std, phenomena

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
    colors = ["firebrick", "royalblue", "green", "red", "purple", "yellow", "blue"]
    fig = go.Figure()
    for i,group in enumerate(groups):
        for metric in metrics_names:
            fig.add_trace(go.Scatter(x=phenomena, y=group[metric],mode='lines',name=group_labels[i]+" - "+metric, 
                                     line=dict(color=colors[i])))
    fig.update_layout(
        title=" ".join(group_labels)
    )
    fig.show()


