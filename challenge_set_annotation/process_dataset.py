import datasets
# import pprint
from datasets import load_from_disk
from quantulum3 import parser
import difflib, re

import numpy as np
np.random.seed(42)
import random
import argparse, os, json, copy
from tqdm import tqdm

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

# given a sentence, return the tokens and their start and end indices
def tokenize(sentence):
    s0 = sentence.lower()
    s1 = re.sub(r"[\"\[\]\.,!?:;'\(\)$“„”]+\s", ' ', s0)
    s2 = re.sub(r"\s[\"\[\]\.,!?:;'\(\)$“„”]+",' ', s1)
    s3 = s2.strip("\"[].,!?:;'\(\)$")
    tokenized = s3.split()
    
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
        return ref, r, r_spans
    else:
        return good, g, g_spans

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
        if len(change) != 1:
            logger.error('in hallucination-dates: after removing the abbr. there are %s changes %s, %s'%(len(change), good, bad))
    return change
    

folder = os.getcwd()
dataset_path = os.path.join(folder, 'dataset')
if not os.path.exists(dataset_path):
    logger.error('No dataset path: %s' %(dataset_path))
    exit()

logger.info('Loading the dataset...')
dataset = load_from_disk(dataset_path)
logger.info('Dataset loaded.')

# this is the list of phenomena and which option they need to be annotated with:
phenomena = {
    'addition':'add-omit',
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
    'copy-source':'?',
    'coreference-based-on-commonsense':'mixed_flexible',
    'do-not-translate':'diff_flexible',
    'hallucination-date-time':'date',
    'hallucination-named-entity-level-1':'diff_flexible',
    'hallucination-named-entity-level-2':'REF_flexible',
    'hallucination-named-entity-level-3':'REF_flexible',
    'hallucination-number-level-1':'diff_flexible',
    'hallucination-number-level-2':'REF_flexible',
    'hallucination-number-level-3':'REF_flexible',
    'hallucination-real-data-vs-ref-word':'diff_flexible',
    'hallucination-real-data-vs-synonym':'diff_flexible',
    'hallucination-unit-conversion-amount-matches-ref':'units',
    'hallucination-unit-conversion-unit-matches-ref':'units',
    'hypernym-replacement':'REF_flexible',
    'hyponym-replacement':'REF_flexible',
    'lexical-overlap':'?',
    'modal_verb:deletion':'add-omit',
    'modal_verb:substitution':'diff_flexible',
    'nonsense':'REF_flexible',
    'omission':'add-omit',
    'ordering-mismatch':'swap',
    'overly-literal-vs-correct-idiom':'diff_flexible',
    'overly-literal-vs-explanation':'diff_flexible',
    'overly-literal-vs-ref-word':'diff_flexible',
    'overly-literal-vs-synonym':'diff_flexible',
    'pleonastic_it:deletion':'annotate_word',
    'pleonastic_it:substitution':'annotate_word',
    'punctuation:deletion_all':'add-omit',
    'punctuation:deletion_commas':'add-omit',
    'punctuation:deletion_quotes':'add-omit',
    'punctuation:statement-to-question':'add-omit',
    'real-world-knowledge-entailment':'diff_flexible',
    'real-world-knowledge-hypernym-vs-distractor':'diff_flexible',
    'real-world-knowledge-hypernym-vs-hyponym':'diff_flexible',
    'real-world-knowledge-synonym-vs-antonym':'diff_flexible',
    'similar-language-high':'?',
    'similar-language-low':'?',
    'untranslated-vs-ref-word':'diff_flexible',   # here add-omit can be used for getting character level replacements too
    'untranslated-vs-synonym':'diff_flexible',
    'xnli-addition-contradiction':'?',
    'xnli-addition-neutral':'?',
    'xnli-omission-contradiction':'?',
    'xnli-omission-neutral':'?'
}

# if there are already some annotations overwrite them and append new ones
annotated_dataset_path = os.path.join(folder, 'annotated.txt')
if os.path.exists(annotated_dataset_path):
    logger.info('Path {} already exists. Loading..'.format(annotated_dataset_path))
    with open(annotated_dataset_path, "r") as f:
        annotations = json.load(f)
else:
    annotations = dict()

# calculate statistics about the annotations:
# for every mode, calculate no. of skipped, no. of unsure and ids, and no. of done.
stats_template = {
            'total':0,
            'success':0,
            'too_long':[],
            'no_change':[],
            'error':[],
            'other':[]  
        }
stats = {}
for key in phenomena.keys():
    stats[key] = copy.deepcopy(stats_template)
    
annotations = {}
logger.setLevel(logging.ERROR)
for idx,sample in tqdm(enumerate(dataset["train"])):
    # if sample["phenomena"] in phenomena.keys() and 'Le garçon voulait mettre le jeu dans la boîte, mais c\'était trop grand.' in sample['good-translation']:
    if sample["phenomena"] in phenomena.keys():
        stats[sample["phenomena"]]["total"] += 1
        if phenomena[sample["phenomena"]] == 'add-omit':
            try:
                change = diff_char_level(sample["good-translation"], sample["incorrect-translation"])
                if len(change) == 0:
                    logger.warning('No change in id {}'.format(idx))
                    stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                else:
                    stats[sample["phenomena"]]["success"] += 1
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample
            except:
                logger.warning('error in char level annotate, id {}'.format(idx))
                stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
                
        elif phenomena[sample["phenomena"]] == 'annotate_word':
            try:
                change = annotate_word(sample["good-translation"], sample["incorrect-translation"])
                if len(change) == 0:
                    logger.warning('No change in id {}'.format(idx))
                    stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                else:
                    stats[sample["phenomena"]]["success"] += 1
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample
            except:
                logger.warning('error in word level annotate, id {}'.format(idx))
                stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))

        elif phenomena[sample["phenomena"]] in ['diff_flexible', 'REF_flexible', 'mixed_flexible']:
            if phenomena[sample["phenomena"]] == 'diff_flexible':
                good = sample["good-translation"]
            elif phenomena[sample["phenomena"]] == 'mixed_flexible':
                good, g, g_spans = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"])
            else: 
                good = sample["reference"]
            bad = sample["incorrect-translation"]
            g, g_spans = tokenize(good)
            b, b_spans = tokenize(bad)

            # special treatment to japanese chinese and thailandish because they don't use spaces, so can't be split            
            if sample['langpair'][-2:] not in ['ja', 'zh', 'th']:      
                if len(g) == len(b):   # if there are multiple one word replacements
                    change = diff(g, g_spans, b, b_spans, phenomena="replacement")
                if len(g) != len(b) or len(change) == 0:
                    try:
                        change = diff_flexible(good, g, g_spans, bad, b, b_spans)
                        if len(change) == 0 and good != bad:
                            change = diff_char_level(good, bad) 
                    except:
                        logger.warning('error in id {}'.format(idx))
                        stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
                        continue
                if len(change) == 0:
                    logger.warning('No change in id {}'.format(idx,g,b,change))
                    stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                elif len(change) != 0 and ((change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 50) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 50)):
                    logger.warning('check this - too long: %s' %idx)
                    stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
                else:
                    stats[sample["phenomena"]]["success"] += 1
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample  
            else:
                try:
                    change = diff_char_level(good, bad) 
                    if len(change) == 0 and good != bad:
                        logger.warning('No change in id {}'.format(idx,g,b,change))
                        stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                    elif len(change) != 0 and ((change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 30) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 30)):
                        logger.warning('check this - too long: %s' %idx)
                        stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
                    else:
                        stats[sample["phenomena"]]["success"] += 1
                    sample['annotation'] = change
                    sample['method'] = phenomena[sample["phenomena"]]
                    annotations[idx] = sample
                except: 
                    logger.warning('error in id {}'.format(idx))
                    stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
            
                
        elif phenomena[sample["phenomena"]] == 'units':
            try:
                g, b, change = annotate_units(sample["good-translation"],sample["incorrect-translation"])
                if len(change) == 0 and g != b:
                    logger.warning('No change in id {}, \ng: {}, \nb: {},\nr: {}'.format(idx, g, b))
                    stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                elif len(change) > 1:
                    logger.warning('Multiple changes in {} id {}'.format(sample["phenomena"], idx))
                    stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))
                else:
                    stats[sample["phenomena"]]["success"] += 1
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample  
            except: 
                logger.warning('error in id {}'.format(idx))
                stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
            
        elif phenomena[sample["phenomena"]] == 'swap':
            try:
                change = annotate_swap_word_lvl(sample["good-translation"],sample["incorrect-translation"])
                if len(change) < 2 and sample["good-translation"] != sample["incorrect-translation"]:
                    logger.warning('No change in id {}, \ng: {}, \nb: {}'.format(idx, sample["good-translation"], sample["incorrect-translation"]))
                    stats[sample["phenomena"]]["no_change"].append((idx, sample['langpair']))
                elif change[0]['in_good'] != None and change[1]['in_good'] != None and change[0]['in_good'] == change[1]['in_good']:
                    logger.warning('check this: %s - swapped words are the same!' %idx)
                    stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))
                elif (change[0]['in_good'] != None and len(change[0]['in_good']['token']) > 50) or (change[0]['in_bad'] != None and len(change[0]['in_bad']['token']) > 50):
                    logger.warning('check this: %s' %idx)
                    stats[sample["phenomena"]]["too_long"].append((idx, sample['langpair']))
                else:
                    stats[sample["phenomena"]]["success"] += 1
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample
            except: 
                logger.warning('error in id {}'.format(idx))
                stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
            
        elif phenomena[sample["phenomena"]] == 'date':
            try:
                change = diff_dates(sample["good-translation"],sample["incorrect-translation"])
                stats[sample["phenomena"]]["success"] += 1
                sample['annotation'] = change
                sample['method'] = phenomena[sample["phenomena"]]
                annotations[idx] = sample
            except: 
                logger.warning('error in id {}'.format(idx))
                stats[sample["phenomena"]]["error"].append((idx, sample['langpair']))
                
        # else:
        #    stats[sample["phenomena"]]["other"].append((idx, sample['langpair']))

with open(annotated_dataset_path, "w") as f:
    json.dump(annotations, f)  # encode dict into JSON
stats_path = os.path.join(folder, 'stats.txt')
with open(stats_path, "w") as f:
    json.dump(stats, f)  # encode dict into JSON
logger.info("Done writing dict into {} file".format(annotated_dataset_path))