#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from quantulum3 import parser
import difflib, re, copy
import numpy as np
np.random.seed(42)

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from sacremoses import MosesPunctNormalizer
from sacremoses import MosesTokenizer, MosesDetokenizer

# given a sentence, return the tokens and their start and end indices
def tokenize(sentence):
    s0 = sentence.lower()
    s0 = re.sub('i̇', 'i', s0)
    s = re.sub(r"[\"\[\]\.,!?:;'\(\)$“„”]+\s", ' ', s0)
    s = re.sub(r"^[\"\[\]\.,!?:;'\(\)$“„”]+", ' ', s)
    s = re.sub(r"\s[\"\[\]\.,!?:;'\(\)$“„”]+",' ', s)
    s = re.sub(r"[\"\[\]\.,!?:;'\(\)$“„”]+$",' ', s)
    s = s.strip("\"[].,!?:;'\(\)$")
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
def ref_or_good(ref, good, bad):
    g, g_spans = tokenize(good)
    b, b_spans = tokenize(bad)
    r, r_spans = tokenize(ref)
    g_change = diff_flexible(good, g, g_spans, bad, b, b_spans)
    r_change = diff_flexible(ref, r, r_spans, bad, b, b_spans)
    if len(r_change[0]["in_good"]["token"]) <= len(g_change[0]["in_good"]["token"]):
        return ref
    else:
        return good

# Span annotations for the addition data - word ids for now
# can handle multiple replacements, only one adddition and omission
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
     
def annotate_word(good, incorrect):
    g, g_spans = tokenize(good)
    b, b_spans = tokenize(incorrect)
    try:
        if len(g) > len(b):    # omission
            # return diff(b, g)
            return diff(g, g_spans, b, b_spans, phenomena="omission")

        elif len(g) < len (b):   # addition
            return diff(g, g_spans, b, b_spans, phenomena="addition")

        else:   # replacement
            return diff(g, g_spans, b, b_spans, phenomena="replacement")

    except:
        logging.error("Error in annotate!")
        return None

# find addition, omission and SINGLE replacements but directly annotates on the characater level now
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
def annotate_units(good,bad):
    units_g = [u.surface for u in parser.parse(good)]
    units_b = [u.surface for u in parser.parse(bad)]
    i = 0
    g, g_spans = tokenize(good)
    b, b_spans = tokenize(bad)
    changes = []
    for i in range(len(units_g)):
        logger.debug([units_g[i], i])
        if units_g[i] != units_b[i]:
            ref_pattern = units_g[i].split()[0]
            logger.debug('ref_pattern: ' + ref_pattern)
            for pid in range(len(g)-len(units_g[i].split())+1):
                logger.debug([g[pid], g[pid] == ref_pattern])
                if ref_pattern in g[pid]:
                    start = pid
            g_tokens = list(range(start,start+len(units_g[i].split())))
            b_tokens = list(range(start,start+len(units_b[i].split())))
            changes.append({"in_good":{'token_index':g_tokens, 
                    'character_span':(g_spans[g_tokens[0]][0], g_spans[g_tokens[-1]][1]), 'token':units_g[i]}, 
                         "in_bad":{'token_index':b_tokens, 
                    'character_span':(b_spans[b_tokens[0]][0], b_spans[b_tokens[-1]][1]), 'token':units_b[i]}})
    return g, b, changes

# find two substrings which are swapped. Maybe later get rid of diff_flexible and make it complately character level?
def annotate_swap_word_lvl(good, bad):
    changes = []
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
    g, g_spans = tokenize(good)
    b, b_spans = tokenize(bad)
    if len(g) != len(b):
        logger.error('in hallucination-dates: %s, %s'%(g,b))
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
def standardize_annotation(change, good, bad, maps=None, original=None):
    change_tmp = copy.deepcopy(change)
    if maps != None:
        good_mapping, bad_mapping = maps[0], maps[1]
        good_og, bad_og = original[0], original[1]
        for c in change_tmp:
            if c["in_good"] != None:
                c["in_good"]['character_span'] = (good_mapping[c["in_good"]['character_span'][0]], good_mapping[c["in_good"]['character_span'][1]])
                c["in_good"]['token'] = good_og[c["in_good"]['character_span'][0]:c["in_good"]['character_span'][1]]
            if c["in_bad"] != None:
                c["in_bad"]['character_span'] = (bad_mapping[c["in_bad"]['character_span'][0]], bad_mapping[c["in_bad"]['character_span'][1]])
                c["in_bad"]['token'] = bad_og[c["in_bad"]['character_span'][0]:c["in_bad"]['character_span'][1]]
        
    skip = False
    for c in change:
        if c['in_good'] == None or c['in_bad'] == None \
        or c['in_good']['token_index'] == None or (type(c['in_good']['token_index'])==list and len(c['in_good']['token_index']) > 1)\
        or c['in_bad']['token_index'] == None or (type(c['in_bad']['token_index'])==list and len(c['in_bad']['token_index']) > 1):
            skip = True
            logger.debug("first check")
            break
    if skip:   # if skipping then change all the integer token indices to lists
        for c in change:
            if c['in_good'] != None and c['in_good']['token_index'] != None and type(c['in_good']['token_index']) != list:
                c['in_good']['token_index'] = [c['in_good']['token_index']]
            if c['in_bad'] != None and c['in_bad']['token_index'] != None and type(c['in_bad']['token_index']) != list:
                c['in_bad']['token_index'] = [c['in_bad']['token_index']]
        return change
    good_tokens = []
    bad_tokens = []
    good_span = ()   # char span
    bad_span = ()
    change_new = []
    for c in change:
        g = c['in_good']
        b = c['in_bad']
        if type(g['token_index']) == list:
            g['token_index'] = g['token_index'][0]
        if type(b['token_index']) == list:
            b['token_index'] = b['token_index'][0]
        if len(good_tokens) == 0 and len(bad_tokens) == 0:
            good_tokens.append(g['token_index'])
            bad_tokens.append(b['token_index'])
            good_span = g['character_span']
            bad_span = b['character_span']
        elif g['token_index'] == good_tokens[-1] + 1 and b['token_index'] == bad_tokens[-1] + 1:
            good_tokens.append(g['token_index'])
            bad_tokens.append(b['token_index'])
            good_span = (good_span[0], g['character_span'][1])
            bad_span = (bad_span[0], b['character_span'][1])
        else:
            change_new.append({'in_good': {'token_index': good_tokens,
                        'character_span': good_span,
                        'token': good[good_span[0]:good_span[1]]}, 
                    'in_bad': {'token_index': bad_tokens,
                        'character_span': bad_span,
                        'token': bad[bad_span[0]:bad_span[1]]}})
            good_tokens = [g['token_index']]
            bad_tokens = [b['token_index']]
            good_span = g['character_span']
            bad_span = b['character_span']

    change_new.append({'in_good': {'token_index': good_tokens,
                        'character_span': good_span,
                        'token': good[good_span[0]:good_span[1]]}, 
                    'in_bad': {'token_index': bad_tokens,
                        'character_span': bad_span,
                        'token': bad[bad_span[0]:bad_span[1]]}})
    return change_new

# return detokenized sentence, and the ids of the removed spaces
# or mapping for each char from detokenized sentence to original?
def detokenize_text(sentence, lang='en'):
    mt, md = MosesTokenizer(lang=lang), MosesDetokenizer(lang=lang)
    mpn = MosesPunctNormalizer()
    detokenized = md.detokenize(mt.tokenize(md.detokenize(mpn.normalize(sentence).split())))
    logger.debug("detokenized: {}".format(detokenized))
    mapping = dict()
    i = 0
    for d_id in range(len(detokenized)):
        logger.debug("outer: {}".format([detokenized[d_id], d_id, sentence[i], i]))
        while detokenized[d_id] != sentence[i]:
            i += 1
            logger.debug("inner: {}".format([detokenized[d_id], d_id, sentence[i], i]))
        mapping[d_id] = i 
    mapping[len(detokenized)] = len(sentence)       
    return detokenized, mapping

# If same ref and incorrect sentence was annotated before then just copy the annotation
def check_seen_before(sample, annotations):
    for annotated_sample in annotations.values():
        if annotated_sample["reference"] == sample["reference"] and annotated_sample["incorrect-translation"] == sample["incorrect-translation"]:
              return (annotated_sample["annotation"], annotated_sample["method"])
    return None
