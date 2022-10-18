#!/usr/bin/env python3

import logging

from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr

import pandas as pd

from breakit.scorers import BLEUScorer, CHRFScorer, COMETScorer
from breakit.utils import read_file

SCORERS = {'bleu': BLEUScorer,
           'chrf': CHRFScorer,
           'comet': COMETScorer,
           'comet-qe': COMETScorer}


def get_arg_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser(description='Command for scoring adversarial translation hypotheses.')

    parser.add_argument('-i', '--inputs',
                        type=Path_fr,
                        nargs='+',
                        required=True,
                        help='Input TSV files with sources, translation hypotheses and reference translations.')
    parser.add_argument('-o', '--output_suffix',
                        type=str,
                        default='scored',
                        help='Output suffix to add to processed TSV files.')
    parser.add_argument('-m', '--metrics',
                        type=str,
                        nargs='+',
                        choices=['bleu', 'chrf', 'comet', 'comet-qe'],
                        default=['bleu', 'chrf', 'comet','comet-qe'],
                        help='Metrics to score on examples.')
    parser.add_argument('-p', '--path',
                        type=str,
                        default='wmt20-comet-qe-da',
                        help='Absolute path for metric') # TODO if multiple metrics require multiple paths, add a list corresponding to the order of the metrics that were called




    return parser


def score() -> None:
    '''
    Score all TSV files with all selected metrics and save as new TSV file.
    '''
    cfg = get_arg_parser().parse_args()
    scorers = []
    for m in cfg.metrics:
        if 'comet-qe' in m:
            assert cfg.path is not None
            scorers.append(SCORERS[m](model_path=cfg.path, use_reference=False))
        else:
            scorers.append(SCORERS[m]())


    #scorers = [SCORERS[m]() for m in cfg.metrics]

    for f in cfg.inputs:
        tsv_f = read_file(f)
        logging.info(f'Scoring {f}')

        for scorer in scorers:
            scorer(tsv_f)

        # Write new TSV file
        new_f = f.split('tsv')[0]+cfg.output_suffix+'.tsv'
        tsv_f.to_csv(new_f, index=False, sep='\t')


if __name__ == '__main__':
        score()
