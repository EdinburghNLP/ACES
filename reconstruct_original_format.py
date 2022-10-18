#!/usr/bin/env python3

import logging
import glob
import re
import os

from collections import defaultdict

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd



def get_arg_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser(description='Command for reconstructing input format for breakit-eval.')

    parser.add_argument('-p', '--path',
                        type=str,
                        required=True,
                        help='Path where folders with files are stored.')
    parser.add_argument('-o', '--output_filename',
                        type=str,
                        required=True,
                        help='Output prefix for reconstructed TSV file.')
    parser.add_argument('-l', '--langpairs',
                        type=str,
                        nargs='+',
                        default=[],
                        help='Language pairs for which data should be extracted.')

    return parser


def main() -> None:
    '''
    Extract source, reference, good-translation, incorrect-translation, phenomena and model scores.
    '''
    cfg = get_arg_parser().parse_args()

    tsv_files = []

    rootdir = Path(cfg.path)

    for langpair in rootdir.glob('*'):
        if not re.match(r'\w\w-\w\w', langpair.name):
            continue

        langpair_name = langpair.name.split('/')[-1]

        if cfg.langpairs and langpair_name not in cfg.langpairs:
            continue

        gold_tsv = pd.read_csv(f'{langpair}/gold.tsv', sep='\t', index_col=0, header=0, error_bad_lines=False)

        lang_tsv = pd.DataFrame()

        lang_tsv['source'] = gold_tsv['source']
        lang_tsv['reference'] = gold_tsv['reference']
        lang_tsv['phenomena'] = gold_tsv['phenomena']
        lang_tsv['langpair'] = gold_tsv['langpair']

        models = [m for m in langpair.glob('*/*') if not m.name.startswith('._')]

        print(f'Processing: {langpair} - {len(models)} models')

        for model in models:
            good = []
            bad = []
            metric_good = []
            metric_bad = []
            model_name = re.sub('scores.tsv', '', os.path.basename(model))
            model_tsv = pd.read_csv(model, sep='\t', index_col=0, header=0, error_bad_lines=False)

            for (_, e), (_, m) in zip(gold_tsv.iterrows(), model_tsv.iterrows()):
                score_a = m['SystemA']
                score_b = m['SystemB']
                if type(score_a) == str and score_a.startswith('tensor'):
                    score_a = score_a.lstrip('tensor(')
                    score_a = score_a.rstrip(')')
                if type(score_b) == str and score_b.startswith('tensor'):
                    score_b = score_b.lstrip('tensor(')
                    score_b = score_b.rstrip(')')
                score_a = float(score_a)
                score_b = float(score_b)
                if e['systemA_correct'] == 1:
                    good.append(e['systemA'])
                    bad.append(e['systemB'])
                    metric_good.append(score_a)
                    metric_bad.append(score_b)
                elif e['systemB_correct'] == 1:
                    good.append(e['systemB'])
                    bad.append(e['systemA'])
                    metric_good.append(score_b)
                    metric_bad.append(score_a)
                else:
                    raise ValueError('Both translations incorrect.')

            lang_tsv[f'{model_name}-good'] = metric_good
            lang_tsv[f'{model_name}-bad'] = metric_bad

        lang_tsv['good-translation'] = good
        lang_tsv['incorrect-translation'] = bad

        tsv_files.append(lang_tsv)


    tsv_f = pd.concat(tsv_files)
    tsv_f.to_csv(cfg.output_filename, index=False, sep='\t')



if __name__ == '__main__':
        main()
