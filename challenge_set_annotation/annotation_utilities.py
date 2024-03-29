#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from quantulum3 import parser
import difflib, re, copy, string, json
import numpy as np
np.random.seed(42)

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer, MosesDetokenizer

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
    'do-not-translate':'diff_flexible',
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
    'modal_verb:deletion':'annotate_word',
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
    'xnli-addition-contradiction':'whole_sentence',
    'xnli-addition-neutral':'whole_sentence',
    'xnli-omission-contradiction':'whole_sentence',
    'xnli-omission-neutral':'whole_sentence'
}

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

# Partly Manually annotated ones
manual_ids = ['19572',
 '25385',
 '25400',
 '19597',
 '19603',
 '19610',
 '25367',
 '25371',
 '25381',
 '25383',
 '25388',
 '25390',
 '25393',
 '25399',
 '25405',
 '25417',
 '25422',
 '24607',
 '24612',
 '24654', '16018',
 '16048',
 '16095',
 '16111',
 '16147',
 '32356',
 '32376',
 '32406',
 '32412',
 '32424',
 '32429',
 '32456',
 '16175',
 '16176',
 '16180',
 '16341',
 '16348', '27506', '17252', '17261', '17288', '17390', '35839',
 '35840',
 '35841',
 '35843',
 '35845',
 '35847',
 '35848',
 '35849',
 '35850',
 '35851',
 '35853',
 '35854',
 '35855',
 '35856',
 '35857']

# Define a custom JSON encoder to handle special characters without extra quotation marks
class SpecialCharacterEncoder(json.JSONEncoder):
    def encode(self, obj):
        return super().encode(obj).replace(r'\"', '"')
"""  
class SpecialCharacterDecoder(json.JSONDecoder):
    def decode(self, s):
        return super().decode(s)
"""
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, cls=SpecialCharacterEncoder, ensure_ascii=False, indent=2)

def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        json_str = file.read()
        return json.loads(json_str)

# given a sentence, return the tokens and their start and end indices
# for ko. ja, zh, th: turn each character to a word and tokenize like that. also for those in detokenizer remove all spaces.
def tokenize(sentence, chars=False):
    s = re.sub('i̇', 'i', sentence)
    if chars:
        tokenized = list(s)
        spans = [(i,i+1) for i in range(len(tokenized))]
        return tokenized, spans
    punctuation = string.punctuation
    punctuation += "\"\[\]\.,!?:;'\(\)$“„”"
    s0 = s
    s0 = re.sub(r'(^|[^A-Za-z0-9])([' + re.escape(punctuation) + '])([' + re.escape(punctuation) + '])([' + re.escape(punctuation) + '])(.|$)', r'\1 \2 \3 \4 \5', s0)
    s0 = re.sub(r'(^|[^A-Za-z0-9])([' + re.escape(punctuation) + '])([' + re.escape(punctuation) + '])(.|$)', r'\1 \2 \3 \4', s0)
    s0 = re.sub(r'(^|[^A-Za-z0-9])([' + re.escape(punctuation) + '])(.|$)', r'\1 \2 \3', s0)
    s0 = re.sub(r'(^|.)([' + re.escape(punctuation) + '])([' + re.escape(punctuation) + '])([' + re.escape(punctuation) + '])([^A-Za-z0-9]|$)', r'\1 \2 \3 \4 \5', s0)
    s0 = re.sub(r'(^|.)([' + re.escape(punctuation) + '])([' + re.escape(punctuation) + '])([^A-Za-z0-9]|$)', r'\1 \2 \3 \4', s0)
    s0 = re.sub(r'(^|.)([' + re.escape(punctuation) + '])([^A-Za-z0-9]|$)', r'\1 \2 \3', s0)
    tokenized = s0.split()
    spans = []
    split = 0
    for token in tokenized:
        res = re.search(re.escape(token), s[split:])
        start = res.start() + split
        end = res.end() + split
        spans.append((start, end))
        split = end
    return tokenized, spans

def tokenize_remove_punctuation(sentence, chars=False):
    s0 = re.sub('i̇', 'i', sentence)
    s = re.sub(r"[\"\[\]\.,!?:;'\(\)$“„”]+\s", ' ', s0)
    s = re.sub(r"^[\"\[\]\.,!?:;'\(\)$“„”]+", ' ', s)
    s = re.sub(r"\s[\"\[\]\.,!?:;'\(\)$“„”]+",' ', s)
    s = re.sub(r"[\"\[\]\.,!?:;'\(\)$“„”]+$",' ', s)
    s = s.strip("\"[].,!?:;'\(\)$")
    if chars:
        tokenized = list(s)
        spans = [(i,i+1) for i in range(len(tokenized))]
        return tokenized, spans
    tokenized = s.split()
    spans = []
    split = 0
    for token in tokenized:
        res = re.search(re.escape(token), s0[split:])
        start = res.start() + split
        end = res.end() + split
        spans.append((start, end))
        split = end
    return tokenized, spans


# choose whether the reference sentence or the good sentence was changed to create the incorrect translation.
def ref_or_good(ref, good, bad, chars=False):
    g, g_spans = tokenize(good, chars=chars)
    b, b_spans = tokenize(bad, chars=chars)
    r, r_spans = tokenize(ref, chars=chars)
    g_change = diff_flexible(good, g, g_spans, bad, b, b_spans)
    r_change = diff_flexible(ref, r, r_spans, bad, b, b_spans)
    if len(r_change[0]["in_good"]["token"]) <= len(g_change[0]["in_good"]["token"]):
        return ref
    else:
        return good

# word level Span annotations for replacements, adddition and omission
def diff(g, g_spans, b, b_spans, phenomena="addition"):
    i, j = 0, 0
    change = []
    while j < len(b) and i < len(g):
        logger.debug([g[i], i, b[j], j])
        if g[i] == b[j]:
            i += 1
            j += 1
        else:
            if phenomena == "addition":
                change.append({'in_good': None, 'in_bad': {'token_index':j, 'character_span':b_spans[j], 'token':b[j]}})
                j += 1 
            elif phenomena == "replacement":
                change.append({'in_good': {'token_index':i, 'character_span':g_spans[i], 'token':g[i]},
                               'in_bad': {'token_index':j, 'character_span':b_spans[j], 'token':b[j]}})
                i += 1
                j += 1
            # for omission also return where the deleted part was supposed to be in incorrect translation ?
            elif phenomena == "omission":
                change.append({'in_good': {'token_index':i, 'character_span':g_spans[i], 'token':g[i]},
                               'in_bad': None})
                i += 1
    if phenomena == "addition":
        change.extend([{'in_good': None, 'in_bad': {'token_index':jx, 'character_span':b_spans[jx], 'token':b[jx]}} 
                       for jx in range(j,len(b))])
    elif phenomena == "omission":
        change.extend([{'in_good': {'token_index':ix, 'character_span':g_spans[ix], 'token':g[ix]},
                               'in_bad': None} for ix in range(i,len(g))])
    return change
     
def annotate_word(good, incorrect, chars=False):
    if good.lower() != incorrect.lower():
        g, g_spans = tokenize(good.lower(), chars=chars)
        b, b_spans = tokenize(incorrect.lower(), chars=chars)
    else:
        g, g_spans = tokenize(good, chars=chars)
        b, b_spans = tokenize(incorrect, chars=chars)
    try:
        if len(g) > len(b):    # omission
            return diff(g, g_spans, b, b_spans, phenomena="omission")

        elif len(g) < len (b):   # addition
            return diff(g, g_spans, b, b_spans, phenomena="addition")

        else:   # replacement
            return diff(g, g_spans, b, b_spans, phenomena="replacement")

    except:
        logging.error("Error in annotate!")
        return None

# find addition, omission and SINGLE replacements but directly annotates on the characater level now
def diff_char_level(good, bad):
    s = difflib.SequenceMatcher(lambda x: x == "", good, bad, autojunk=False)
    good_extra = list()
    bad_extra = list()
    g_lim = 0
    b_lim = 0
    for block in s.get_matching_blocks():
        a,b,size = block
        logger.debug("good[%s:%s] and bad[%s:%s] match for %s characters: %s" % (a, a+size, b, b+size, size, good[a:a+size]))
        if a > g_lim:
            good_extra.append((g_lim, a, good[g_lim:a]))
        g_lim = a + size
        if b > b_lim:
            bad_extra.append((b_lim, b, bad[b_lim:b]))
        b_lim = b + size
    logger.debug(good_extra)
    logger.debug(bad_extra)

    change = []
    if len(bad_extra) > 0 and len(good_extra) == 0:
        for block in bad_extra:
            change.append({'in_good': None, 'in_bad': {'token_index':None, 'character_span':(block[0],block[1]), 'token':block[2]}})
    elif len(good_extra) > 0 and len(bad_extra) == 0:
        for block in good_extra:
            change.append({'in_good': {'token_index':None, 'character_span':(block[0],block[1]), 'token':block[2]}, 'in_bad': None})
    elif len(bad_extra) > 0 and len(good_extra) > 0:  
        if len(bad_extra) == len(good_extra):
            for i in range(len(bad_extra)):
                b_start, b_end = bad_extra[i][0], bad_extra[i][1]
                g_start, g_end = good_extra[i][0], good_extra[i][1]
                change.append({'in_good': {'token_index':None, 'character_span':(g_start, g_end), 'token':good_extra[i][2]},
                       'in_bad': {'token_index':None, 'character_span':(b_start, b_end), 'token':bad_extra[i][2]}})
        else:
            b_start, b_end = bad_extra[0][0], bad_extra[-1][1]
            g_start, g_end = good_extra[0][0], good_extra[-1][1]
            change.append({'in_good': {'token_index':None, 'character_span':(g_start, g_end), 'token':good[g_start:g_end]},
                       'in_bad': {'token_index':None, 'character_span':(b_start, b_end), 'token':bad[b_start:b_end]}})
        
        # logger.debug("in diff replacement: \ngood: %s, \nbad: %s, \nchange: %s" %(good, bad, change))

    return change

# this finds if any number of words are replaced with any number of words - but only for one span
# if there is more than one span, for example as in
# good = "What does slip ring starter and a b starter mean"
# bad = "What does savory starter and c starter mean"
# then it finds it as one big span: ('slip ring starter and a b', [2, 3, 4, 5, 6, 7], 'savory starter and c', [2, 3, 4, 5])
def diff_flexible(good, g, g_spans, bad, b, b_spans, phenomena="default"):
    i = 0
    change = []
    while i < len(b) and i < len(g) and g[i] == b[i]:
        logger.debug([g[i], i])
        i += 1
    start = i
    if start < len(b) and start < len(g):
        i = len(g) - 1
        j = len(b) - 1
        while i > start and j > start and g[i] == b[j]:
            logger.debug([g[i], i, b[j], j])
            i -= 1
            j -= 1
        logger.debug([i, j, start])
        change.append({"in_good":{'token_index':list(range(start,i+1)), 
                    'character_span':(g_spans[start][0],g_spans[i][1]), 
                                  'token':good[g_spans[start][0]:g_spans[i][1]]}, 
                    "in_bad":{'token_index':list(range(start,j+1)), 
                    'character_span':(b_spans[start][0],b_spans[j][1]), 
                               'token':bad[b_spans[start][0]:b_spans[j][1]]}})          
        
    return change

# this can detect multiple spans, but they don't exist in hallucination-unit-conversion-amount-matches-ref 
# and hallucination-unit-conversion-unit-matches-ref. I assume only the numbers and units are changed, so the starting index
# is same in both good and bad translations. But the length of the units can be different (100 miles -> 100 miles per hour)
# here if there are units with a non-latin script with detokenization problems that could cause problems
def annotate_units(good,bad, mode="Mode not given"):
    units_g = [u for u in parser.parse(good)]
    units_b = [u for u in parser.parse(bad)]
    if good.lower() != bad.lower():
        g, g_spans = tokenize_remove_punctuation(good.lower())
        b, b_spans = tokenize_remove_punctuation(bad.lower())
    else:
        g, g_spans = tokenize_remove_punctuation(good)
        b, b_spans = tokenize_remove_punctuation(bad)
    changes = []
    for i in range(len(units_g)):
        if units_g[i].unit.name != "dimensionless" and units_g[i].surface != units_b[i].surface:
            g_parsed = units_g[i]
            b_parsed = units_b[i]
            g_start = g_parsed.span[0]
            for i,s in enumerate(g_spans):
                if s[0] == g_start:
                    g_value_span = s
                    g_value_name = good[s[0]:s[1]]
                    g_value_token = [i]
                    g_units_len = len(g_parsed.surface.split())-1
                    g_unit_token = list(range(i+1,i+1+g_units_len))
                    g_unit_span = (g_spans[g_unit_token[0]][0], g_spans[g_unit_token[-1]][1])
                    g_unit_name = good[g_unit_span[0]:g_unit_span[1]]
                    
            b_start = b_parsed.span[0]
            for i,s in enumerate(b_spans):
                if s[0] == b_start:
                    b_value_span = s
                    b_value_name = bad[s[0]:s[1]]
                    b_value_token = [i]
                    b_units_len = len(b_parsed.surface.split())-1
                    b_unit_token = list(range(i+1,i+1+b_units_len))
                    b_unit_span = (b_spans[b_unit_token[0]][0], b_spans[b_unit_token[-1]][1])
                    b_unit_name = bad[b_unit_span[0]:b_unit_span[1]]
                    
            # assert good[g_value_span[0]:g_value_span[1]] in str(g_parsed.value) 
            # assert bad[b_value_span[0]:b_value_span[1]] in str(b_parsed.value)
            # assert " ".join(g[token] for token in g_value_token) in str(g_parsed.value)
            # assert " ".join(b[token] for token in b_value_token) in str(b_parsed.value)
            # assert g_parsed.unit.name in  " ".join(g[token] for token in g_unit_token)
            # assert b_parsed.unit.name in " ".join(b[token] for token in b_unit_token)
            # assert g_parsed.unit.name in good[g_unit_span[0]:g_unit_span[1]]
            # assert b_parsed.unit.name in bad[b_unit_span[0]:b_unit_span[1]]
            if mode == 'hallucination-unit-conversion-amount-matches-ref':         # number correct, unit wrong
                g_name = g_unit_name
                g_tokens = g_unit_token
                g_span = g_unit_span
                b_name = b_unit_name
                b_tokens = b_unit_token
                b_span = b_unit_span
            elif mode == 'hallucination-unit-conversion-unit-matches-ref':    # number wrong, unit correct, 
                g_name = g_value_name
                g_tokens = g_value_token
                g_span = g_value_span
                b_name = b_value_name
                b_tokens = b_value_token
                b_span = b_value_span
            else:
                logger.info('?? type: {}'.format(type))

            changes.append({"in_good":{'token_index':g_tokens, 
                    'character_span':g_span, 'token':g_name}, 
                        "in_bad":{'token_index':b_tokens, 
                    'character_span':b_span, 'token':b_name}}) 
        elif units_g[i].unit.name == "dimensionless" and units_g[i].surface != units_b[i].surface:
            g_parsed = units_g[i]
            b_parsed = units_b[i]
            g_start = g_parsed.span[0]
            for i,s in enumerate(g_spans):
                if s[0] == g_start:
                    g_value_span = s
                    g_value_name = good[s[0]:s[1]]
                    g_value_token = [i]
                    
            b_start = b_parsed.span[0]
            for i,s in enumerate(b_spans):
                if s[0] == b_start:
                    b_value_span = s
                    b_value_name = bad[s[0]:s[1]]
                    b_value_token = [i]

            g_name = g_value_name
            g_tokens = g_value_token
            g_span = g_value_span
            b_name = b_value_name
            b_tokens = b_value_token
            b_span = b_value_span
        
            changes.append({"in_good":{'token_index':g_tokens, 
                    'character_span':g_span, 'token':g_name}, 
                        "in_bad":{'token_index':b_tokens, 
                    'character_span':b_span, 'token':b_name}}) 
    return g, b, changes

# find two substrings which are swapped. Maybe later get rid of diff_flexible and make it complately character level?
# what if ko, ja, zh, th? 
def annotate_swap_word_lvl(good, bad):
    changes = []
    if good.lower() != bad.lower():
        g, g_spans = tokenize(good.lower())
        b, b_spans = tokenize(bad.lower())
    else:
        g, g_spans = tokenize(good)
        b, b_spans = tokenize(bad)
    r = diff_flexible(good, g, g_spans, bad, b, b_spans, phenomena='swap')[0]

    start = r["in_good"]["token_index"][0]
    end = r["in_good"]["token_index"][-1]

    g = np.array(g)[r["in_good"]["token_index"]]
    b = np.array(b)[r["in_bad"]["token_index"]]
    logger.debug([g, b, (start, end)])
    w1_g = [g[0]]
    w1_b = [b[-1]]
    logger.debug([w1_g, w1_b])
    i = 1
    while w1_g != w1_b and i < len(g):
        w1_g += [g[i]]
        w1_b = [b[len(b)-1-i]] + w1_b
        i += 1
        logger.debug([w1_g, w1_b])
    w1_g_tokens = list(range(start,start+len(w1_g)))
    w1_b_tokens = list(range(end-len(w1_b)+1,end+1))
                       
    w1_g_span = (g_spans[w1_g_tokens[0]][0], g_spans[w1_g_tokens[-1]][1])
    w1_b_span = (b_spans[w1_b_tokens[0]][0], b_spans[w1_b_tokens[-1]][1])
                                             
    changes.append({"in_good":{'token_index':w1_g_tokens, 
            'character_span':w1_g_span, 
                          'token':good[w1_g_span[0]:w1_g_span[-1]]}, 
         "in_bad":{'token_index':w1_b_tokens, 
            'character_span':w1_b_span, 
                          'token':bad[w1_b_span[0]:w1_b_span[-1]]}})
    # now exact opposite:
    w2_g = [g[-1]]
    w2_b = [b[0]]
    logger.debug([w2_g, w2_b])
    i = 1
    while w2_g != w2_b and i < len(g):
        w2_b += [b[i]]
        w2_g = [g[len(g)-1-i]] + w2_g
        i += 1
        logger.debug([w2_g, w2_b])
    w2_b_tokens = list(range(start,start+len(w2_b)))
    w2_g_tokens = list(range(end-len(w2_g)+1,end+1))
                       
    w2_g_span = (g_spans[w2_g_tokens[0]][0], g_spans[w2_g_tokens[-1]][1])
    w2_b_span = (b_spans[w2_b_tokens[0]][0], b_spans[w2_b_tokens[-1]][1])
                                             
    changes.append({"in_good":{'token_index':w2_g_tokens, 
            'character_span':w2_g_span, 
                          'token':good[w2_g_span[0]:w2_g_span[-1]]}, 
         "in_bad":{'token_index':w2_b_tokens, 
            'character_span':w2_b_span, 
                          'token':bad[w2_b_span[0]:w2_b_span[-1]]}})
    return changes

# find the s spans in both sentences which are swapped.
# I assume both sentences have the same length, and everything other than the swap are same
# not working well
def annotate_swap_char_lvl(good, bad):
    changes = []
    blocks = difflib.SequenceMatcher(lambda x: x == "", good ,bad).get_matching_blocks()

    start = 0
    end = len(good)
    
    # matching blocks function don't match them when the words are swapped but we can cut out the 
    # 3 largest matching blocks (before first word, between two words, after second word) then we are
    # left with 2 swapped words
    
    # remove the first block if it starts from index 0
    if blocks[0].a == 0:
        start = blocks[0].size
        
    # remove last block if it is at the end of the sentence - sometimes the last block has length 0 so ignore them
    last_id = len(blocks)-1
    while blocks[last_id].size == 0:
        last_id -= 1
    if blocks[last_id].a + blocks[last_id].size == len(good):
        end = blocks[last_id].a
    
    g = good[start:end]
    b = bad[start:end]
    
    # now match again
    blocks = difflib.SequenceMatcher(lambda x: x == "", g ,b).get_matching_blocks()

    max_size = 0
    index = 0
    for i, block in enumerate(blocks):
        if block.size > max_size:
            max_size = block.size
            index = i
    
    changes.append({'good-translation_range':(blocks[index].a+start,blocks[index].a+max_size+start),
                   'incorrect-traslation_range':(blocks[index].b+start,blocks[index].b+max_size+start),
                   'string':g[blocks[index].a:blocks[index].a+max_size]})

    g2 = g[:blocks[index].a] + g[blocks[index].a+max_size:]
    b2 = b[:blocks[index].b] + b[blocks[index].b+max_size:]
    logger.debug([g, b])
    logger.debug([g2, b2])
    temp = difflib.SequenceMatcher(None, g2 ,b2)

    blocks = temp.get_matching_blocks()
    max_size = 0
    index = 0
    for i, block in enumerate(blocks):
        if block.size > max_size:
            max_size = block.size
            index = i

    word = g2[blocks[index].a:blocks[index].a+max_size]
    logger.debug('second word; {} size {}'.format(word, max_size))
    logger.debug(blocks[index])
    changes.append({'good-translation_range':(g.index(word)+start,g.index(word)+max_size+start),
                   'incorrect-traslation_range':(b.index(word)+start,b.index(word)+max_size+start),
                   'string':word})
    return changes

# Assuming from the paper: there is only one month name which was changed to another month. 
# reference->good change one month name with its abbreviation.
# good->incorrect change one month name to another.
def diff_dates(good, bad):
    g, g_spans = tokenize_remove_punctuation(good)
    b, b_spans = tokenize_remove_punctuation(bad)
    assert len(g) == len(b)
    change = diff(g, g_spans, b, b_spans, phenomena="replacement") 
    if len(change) > 1: 
        # find one change with minimum overlap between two month names
        # I only consider the continuous substring in the beginning of the words as overlap
        min_size = 100
        min_ch = 0
        logger.debug('in dates change: %s' %change)
        for ch in change:
            block = difflib.SequenceMatcher(lambda x: x == "", ch['in_good']['token'], ch['in_bad']['token']).get_matching_blocks()[0]
            if block.a != 0 or block.b != 0:
                logger.debug('in dates before pruning: %s, %s'%(block.a, block.b))
                return [ch]
            if block.size < min_size:
                min_size = block.size
                min_ch = ch
        return [min_ch]
    return change

def whole_sentence(good, bad):
    change = [{"in_good":{'token_index':list(range(0,len(tokenize(good)[1]))), 
                'character_span':(0,len(good)), 
                            'token':good}, 
            "in_bad":{'token_index':list(range(0,len(tokenize(bad)[1]))), 
                'character_span':(0,len(bad)), 
                            'token':bad}}]
    return change

# if we have multiple word spans next to each other, then concatenate them in one span.
# no need for this when we have token_index: None or token_index:list because then it is already one big span
# also make sure token_index is a list for all changes
def concat_words_helper(stack, in_1, in_2, original):     # if we are processing good translation, in_1:"in_good" and in_2:"in_bad"
    # originalis the good original sentence if we are processing good
    tokens = []
    span = ()   # char span
    change_new = []
    for ann in stack:
        if type(ann['token_index']) == list:
            ann['token_index'] = ann['token_index'][0]
        if len(tokens) == 0:
            tokens.append(ann['token_index'])
            span = ann['character_span']
        elif ann['token_index'] == tokens[-1] + 1:
            tokens.append(ann['token_index'])
            span = (span[0], ann['character_span'][1])
        else:
            change_new.append({in_1: {'token_index': tokens,
                        'character_span': span,
                        'token': original[span[0]:span[1]]}, 
                    in_2: None})
            tokens = [ann['token_index']]
            span = ann['character_span']

    change_new.append({in_1: {'token_index': tokens,
                        'character_span': span,
                        'token': original[span[0]:span[1]]}, 
                    in_2: None})
    return change_new

# apply this function after 1) mapping back to un-detokenized version and 2) making sure all token indices are list
def concat_words(change, originals):
    (good, bad) = originals
    change_new = []
    g_stack = []
    b_stack = []
    for c in change:
        if c['in_good'] != None:
            g_stack.append(c['in_good'])
        if c['in_bad'] != None:
            b_stack.append(c['in_bad'])
    if len(g_stack) > 0:
        change_new.extend(concat_words_helper(g_stack, "in_good", "in_bad", good))
    if len(b_stack) > 0:
        change_new.extend(concat_words_helper(b_stack, "in_bad", "in_good", bad))    
    return change_new

def revert_detokenize(change, good, bad, maps=None, originals=None):
    change_tmp = copy.deepcopy(change)
    if maps != None:
        good_mapping, bad_mapping = maps[0], maps[1]
        good_og, bad_og = originals[0], originals[1]
        good_mapping[len(good)] = len(good_og)
        bad_mapping[len(bad)] = len(bad_og)
        for c in change_tmp:
            if c["in_good"] != None:
                c["in_good"]['character_span'] = (good_mapping[c["in_good"]['character_span'][0]], good_mapping[c["in_good"]['character_span'][1]])
                c["in_good"]['token'] = good_og[c["in_good"]['character_span'][0]:c["in_good"]['character_span'][1]]
            if c["in_bad"] != None:
                c["in_bad"]['character_span'] = (bad_mapping[c["in_bad"]['character_span'][0]], bad_mapping[c["in_bad"]['character_span'][1]])
                c["in_bad"]['token'] = bad_og[c["in_bad"]['character_span'][0]:c["in_bad"]['character_span'][1]]
    return change_tmp

def standardize_annotation(change, good, bad, maps=None, originals=None):
    # check if it is an omission: does not work for moving a word from one place to another, 
    # but we can't annotate that automatically anyway
    omission = False
    for c in change:
        if c["in_good"] != None and c['in_bad'] == None:
            omission = True
    change_reverted = revert_detokenize(change, good, bad, maps, originals)
    change_new = concat_words(change_reverted, originals)
    return change_new, omission

# return detokenized sentence, and the ids of the removed spaces
# or mapping for each char from detokenized sentence to original?
def detokenize_text(sentence, lang='en'):
    if lang in ['ja', 'zh', 'th', 'ko']:
        mpn = MosesPunctNormalizer()
        punct_normalized = mpn.normalize(sentence)
        # punct_normalized = re.sub("\"", "", punct_normalized)
        detokenized = re.sub(' ', '', punct_normalized)
        # d1 = re.sub('(?<=[0-9])\s+(?=[0-9])', '', punct_normalized)
        # detokenized = re.sub('(?<=[a-zA-Z])\s+(?=[a-zA-Z])', '', d1)
    else:
        mt, md = MosesTokenizer(lang=lang), MosesDetokenizer(lang=lang)
        mpn = MosesPunctNormalizer()
        punct_normalized = mpn.normalize(sentence)
        # punct_normalized = re.sub("\"", "", punct_normalized)
        d1 = md.detokenize(mt.tokenize(md.detokenize(punct_normalized.split())))
        d2 = md.detokenize(mt.tokenize(md.detokenize(d1.split())))
        detokenized = md.detokenize(mt.tokenize(md.detokenize(d2.split())))
    logger.debug("detokenized: {}, punc_normalized: {}".format(detokenized, punct_normalized))
    mapping = dict()
    i = 0
    for d_id in range(len(detokenized)):
        logger.debug("outer: {}".format([detokenized[d_id], d_id, punct_normalized[i], i]))
        while detokenized[d_id] != punct_normalized[i]:         
            i += 1
            logger.debug("inner: {}".format([detokenized[d_id], d_id, punct_normalized[i], i]))
        mapping[d_id] = i 
    assert [detokenized[k] for k in mapping.keys()] == [punct_normalized[v] for v in mapping.values()]
    return detokenized, mapping

# If same ref and incorrect sentence was annotated before then just copy the annotation
def check_seen_before(sample, annotations):
    for idx, annotated_sample in annotations.items():
        if annotated_sample["reference"] == sample["reference"] and annotated_sample["incorrect-translation"] == sample["incorrect-translation"]:
              return (annotated_sample["annotation"], annotated_sample["method"]), idx
    return None