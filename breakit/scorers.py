#!/usr/bin/env python3

import pandas as pd

from sacrebleu import sentence_bleu, sentence_chrf
from comet import download_model, load_from_checkpoint


class Scorer(object):

    def __call__(self, tsv_f: pd.DataFrame) -> None:
        raise NotImplementedError


class BLEUScorer(Scorer):

    def __call__(self, tsv_f: pd.DataFrame) -> None:
        tsv_f['bleu-good'] = [sentence_bleu(hyp, [ref]).score for hyp, ref
                              in zip(tsv_f['good-translation'],
                                     tsv_f['reference'])]
        tsv_f['bleu-bad'] = [sentence_bleu(hyp, [ref]).score for hyp, ref
                             in zip(tsv_f['incorrect-translation'],
                                    tsv_f['reference'])]


class CHRFScorer(Scorer):

    def __call__(self, tsv_f: pd.DataFrame) -> None:
        tsv_f['chrf-good'] = [sentence_chrf(hyp, [ref]).score for hyp, ref
                              in zip(tsv_f['good-translation'],
                                     tsv_f['reference'])]
        tsv_f['chrf-bad'] = [sentence_chrf(hyp, [ref]).score for hyp, ref
                             in zip(tsv_f['incorrect-translation'],
                                    tsv_f['reference'])]


class COMETScorer(Scorer):

    def __init__(self,
                 model_path: str='wmt20-comet-da',
                 use_reference: bool=True,
                 gpus: int=1,
                 batch_size: int=16) -> None:
        self.prefix= 'comet'
        if not use_reference:
            assert 'qe' in model_path
            self.prefix = 'cometQE'
        model_path = download_model(model_path)
        self.model = load_from_checkpoint(model_path)
        self.gpus = gpus
        self.batch_size = batch_size

    def __call__(self, tsv_f: pd.DataFrame) -> None:
        data = {'src': tsv_f['source'],
                'mt': tsv_f['good-translation'],
                'ref': tsv_f['reference']}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        tsv_f[self.prefix+'-good'], _ = self.model.predict(data,
                                                    gpus=self.gpus,
                                                    batch_size=self.batch_size)

        data = {'src': tsv_f['source'],
                'mt': tsv_f['incorrect-translation'],
                'ref': tsv_f['reference']}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        tsv_f[self.prefix+'-bad'], _ = self.model.predict(data,
                                                   gpus=self.gpus,
                                                   batch_size=self.batch_size)
