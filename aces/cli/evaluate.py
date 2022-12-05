#!/usr/bin/env python3

import re
from typing import List
from collections import defaultdict

from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr

import pandas as pd
import numpy as np

from aces.utils import read_file


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


def get_arg_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser(description='Command for evaluating different metrics.')

    parser.add_argument('-i', '--inputs',
                        type=Path_fr,
                        nargs='+',
                        required=True,
                        help='Input TSV files with sources, translation hypotheses, reference translations and metric scores.')

    parser.add_argument('-p', '--pretty_print',
                        action='store_true',
                        help='If set will print readable format, otherwise will print format that can be copied to spreadsheet.')

    parser.add_argument('--print_overview',
                        action='store_true',
                        help='If set will print high-level class results.')

    parser.add_argument('--print_aces_score',
                        action='store_true',
                        help='If set will print ACES score.')

    return parser


def comp_aces_score(overview_results: List[float]) -> float:
    '''
    Compute weighted summary score based on top-level correlation scores.
    '''
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

def comp_corr(good_scores: List[float],
              incorrect_scores: List[float]) -> float:
    '''
    Compute correlation as Kendall Tau-like scores.
    '''
    if good_scores.isnull().values.any() or incorrect_scores.isnull().values.any():
        return np.nan

    concordant = np.sum(good_scores > incorrect_scores)
    discordant = np.sum(good_scores <= incorrect_scores)

    tau = (concordant - discordant) / (concordant + discordant)

    return tau


def evaluate() -> None:
    '''
    Evaluate all models in all TSV files and print Kendall Tau score.
    '''
    cfg = get_arg_parser().parse_args()

    for f in cfg.inputs:
        tsv_f = read_file(f)
        overview_results = defaultdict(lambda: defaultdict(list))

        metrics = [re.sub('-good', '', m) for m in tsv_f.columns if m.endswith('-good')]

        # Compute Kendall Tau grouped by phenomenon and metric
        if cfg.pretty_print:
            print(f'\n\nEvaluating {f}')
        phenomena_dfs = [(l, df) for l, df in tsv_f.groupby('phenomena')]

        if not cfg.pretty_print:
            print('phenomenon\t'+'num_examples'+'\t'+'\t'.join(metrics))

        for label, p in phenomena_dfs:
            top_level_label = PHENOMENA_MAPPING[label]
            results = []
            if cfg.pretty_print:
                print(f'\n{label}\t{len(p)}')
            for m in metrics:
                tau = comp_corr(p[f'{m}-good'], p[f'{m}-bad'])
                results.append(str(tau))
                overview_results[m][top_level_label].append(tau)
                if cfg.pretty_print:
                    print(f'\t{m}\t{tau}')
            if not cfg.pretty_print:
                print(label+'\t'+str(len(p))+'\t'+'\t'.join(results))

        if cfg.print_overview:
            print('\nResults from top-level analysis:\n')
            for m in metrics:
                print(f'\n{m}')
                for top_level_label, scores in overview_results[m].items():
                    print(f'\t{top_level_label}\t{np.mean(scores)}')

        if cfg.print_aces_score:
            print('\nSummary score results:\n')
            for m in metrics:
                aces_score = comp_aces_score(overview_results[m])
                print(f'{m}\tACES-score: {aces_score}')

if __name__ == '__main__':
        evaluate()
