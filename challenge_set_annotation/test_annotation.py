#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from annotation_utilities import *
import unittest

class Test(unittest.TestCase):
    # when instead of swapping two words, a word is moved to another place
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

    # word level annotation is not working because when words are swapped a becomes an
    def test_swap_subwords(self):
        sample = {'source': 'Das Beibehalten eines gesunden Energiegleichgewichts, effektive Trinkgewohnheiten und das Verständnis der verschiedenen Aspekte der Einnahme von Ergänzungspräparaten können Athleten helfen, ihre Leistung zu verbessern und ihre Freude am Sport zu steigern.',
 'good-translation': 'Maintaining a healthy energy balance, effective drinking habits, and understanding the different aspects of taking supplements can help athletes improve their performance and increase their enjoyment of sports.',
 'incorrect-translation': 'Maintaining an effective energy balance, healthy drinking habits, and understanding the different aspects of taking supplements can help athletes improve their performance and increase their enjoyment of sports.',
 'reference': 'Maintaining a healthy energy balance, practicing effective hydration habits, and understanding the various aspects of supplementation practices can help athletes improve their performance and increase their enjoyment of the sport.',
 'phenomena': 'ordering-mismatch',
 'langpair': 'de-en'}
        expected = [{'in_good': {'token_index': [1, 2],
    'character_span': (12, 21),
    'token': 'a healthy'},
   'in_bad': {'token_index': [5],
    'character_span': (41, 48),
    'token': 'healthy'}},
  {'in_good': {'token_index': [5],
    'character_span': (38, 47),
    'token': 'effective'},
   'in_bad': {'token_index': [1, 2],
    'character_span': (12, 24),
    'token': 'an effective'}}]
        result = annotate_swap_word_lvl(sample["good-translation"], sample["incorrect-translation"])
        self.assertEqual(result, expected)
        
        # word level annotation is not working because when words are swapped a becomes an
    def test_diff_flexible(self):
        sample = {'source': 'Das Beibehalten eines gesunden Energiegleichgewichts, effektive Trinkgewohnheiten und das Verständnis der verschiedenen Aspekte der Einnahme von Ergänzungspräparaten können Athleten helfen, ihre Leistung zu verbessern und ihre Freude am Sport zu steigern.',
 'good-translation': 'Maintaining a healthy energy balance, effective drinking habits, and understanding the different aspects of taking supplements can help athletes improve their performance and increase their enjoyment of sports.',
 'incorrect-translation': 'Maintaining an effective energy balance, healthy drinking habits, and understanding the different aspects of taking supplements can help athletes improve their performance and increase their enjoyment of sports.',
 'reference': 'Maintaining a healthy energy balance, practicing effective hydration habits, and understanding the various aspects of supplementation practices can help athletes improve their performance and increase their enjoyment of the sport.',
 'phenomena': 'ordering-mismatch',
 'langpair': 'de-en'}
        expected = [{'in_good': {'token_index': [1, 2],
    'character_span': (12, 21),
    'token': 'a healthy'},
   'in_bad': {'token_index': [5],
    'character_span': (41, 48),
    'token': 'healthy'}},
  {'in_good': {'token_index': [5],
    'character_span': (38, 47),
    'token': 'effective'},
   'in_bad': {'token_index': [1, 2],
    'character_span': (12, 24),
    'token': 'an effective'}}]
        result = annotate_swap_word_lvl(sample["good-translation"], sample["incorrect-translation"])
        self.assertEqual(result, expected)
        
        
        
        
if __name__ == '__main__':
    unittest.main()