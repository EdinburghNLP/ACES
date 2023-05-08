#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from annotation_utilities import *
import unittest

class Test(unittest.TestCase):
    def test_swap(self):
        sample = {'source': "Ottawa is Canada's charming, bilingual capital and features an array of art galleries and museums that showcase Canada's past and present.",
 'good-translation': 'Ottawa ist Kanadas charmante, zweisprachige Hauptstadt und bietet eine Reihe von Kunstgalerien und Museen, die Kanadas Vergangenheit und Gegenwart zeigen.',
 'incorrect-translation': 'Ottawa ist Kanadas charmante Hauptstadt und bietet eine Reihe von zweisprachigen Kunstgalerien und Museen, die Kanadas Vergangenheit und Gegenwart zeigen.',
 'reference': 'Ottawa ist Kanadas bezaubernde, zweisprachige Hauptstadt und zeichnet sich durch eine Reihe von Kunstgalerien und Museen aus, die Kanadas Vergangenheit und Gegenwart präsentieren.',
 'phenomena': 'ordering-mismatch',
 'langpair': 'en-de'}
        expected = [{'in_good': {'token_index': [4],
    'character_span': (30, 43),
    'token': 'zweisprachige'},
   'in_bad': {'token_index': [10],
    'character_span': (66, 80),
    'token': 'zweisprachigen'}}]
        result = annotate_swap_word_lvl(sample["good-translation"], sample["incorrect-translation"])
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()