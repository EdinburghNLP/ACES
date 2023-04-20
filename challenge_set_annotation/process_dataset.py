import datasets
# import pprint
from datasets import load_from_disk
from quantulum3 import parser
import difflib, re

import numpy as np
np.random.seed(42)
import random
import argparse, os, json
from tqdm import tqdm

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

# given tokenize a sentence, return the tokens and their start and end indices
def tokenize(sentence):
    s1 = re.sub(r"[\"\[\]\.,!?:;'\(\)$]+\s", ' ', sentence)
    s2 = re.sub(r"\s[\"\[\]\.,!?:;'\(\)$]+",' ', s1)
    s3 = s2.strip("\"[].,!?:;'\(\)$")
    tokenized = s3.split()
    
    spans = []
    split = 0
    for token in tokenized:
        res = re.search(re.escape(token), sentence[split:])
        start = res.start() + split
        end = res.end() + split
        spans.append((start, end))
        split = end
    return tokenized, spans

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
     
def annotate(good, incorrect):
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
                    'character_span':(g_spans[start][0],g_spans[i+1][1]), 
                                  'token':good[g_spans[start][0]:g_spans[i+1][1]]}, 
                    "in_bad":{'token_index':list(range(start,j+1)), 
                    'character_span':(b_spans[start][0],b_spans[j+1][1]), 
                               'token':bad[b_spans[start][0]:b_spans[j+1][1]]}})          
        
    return change

# this can detect multiple spans, but they don't exist in hallucination-unit-conversion-amount-matches-ref 
# and hallucination-unit-conversion-unit-matches-ref. I assume only the numbers and units are changed, so the starting index
# is same in both good and bad translations. But the length of the units can be different (100 miles -> 100 miles per hour)
def annotate_units(g,b):
    units_g = [u.surface for u in parser.parse(g)]
    units_b = [u.surface for u in parser.parse(b)]
    i = 0
    gid, bid = 0, 0
    g_split = g.split()
    b_split = b.split()
    change = []
    for i in range(len(units_g)):
        logger.debug([units_g[i], i])
        if units_g[i] != units_b[i]:
            ref_pattern = units_g[i].split()[0]
            logger.debug('ref_pattern: ' + ref_pattern)
            for pid in range(len(g_split)-len(units_g[i].split())+1):
                logger.debug([g_split[pid], g_split[pid] == ref_pattern])
                if ref_pattern in g_split[pid]:
                    start = pid

            change.append((units_g[i], list(range(start,start+len(units_g[i].split()))), units_b[i], list(range(start,start+len(units_b[i].split())))))
    return g, b, change

def annotate_swap(g, b):
    changes = []
    r = diff_flexible(g.split(),b.split(), g.split(), phenomena='swap')
    prev_len = r[1]+1
    g = r[0][0][0]
    b = r[0][0][2]

    temp = difflib.SequenceMatcher(None, g ,b)

    blocks = temp.get_matching_blocks()
    max_size = 0
    index = 0
    for i, block in enumerate(blocks):
        if block.size > max_size:
            max_size = block.size
            index = i

    changes.append({'good-translation_range':(blocks[index].a+prev_len,blocks[index].a+max_size+prev_len),
                   'incorrect-traslation_range':(blocks[index].b+prev_len,blocks[index].b+max_size+prev_len),
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
    changes.append({'good-translation_range':(g.index(word)+prev_len,g.index(word)+max_size+prev_len),
                   'incorrect-traslation_range':(b.index(word)+prev_len,b.index(word)+max_size+prev_len),
                   'string':word})

    return changes

folder = os.getcwd()
dataset_path = os.path.join(folder, 'dataset')
if not os.path.exists(dataset_path):
    logger.error('No dataset path: '.format(dataset_path))
    exit()

logger.info('Loading the dataset...')
dataset = load_from_disk(dataset_path)
logger.info('Dataset loaded.')

phenomena = {
    'addition':'simple annotate',
    'ambiguous-translation-wrong-discourse-connective-since-causal':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-since-temporal':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-while-contrast':'diff_flexible',
    'ambiguous-translation-wrong-discourse-connective-while-temporal':'diff_flexible',
    'ambiguous-translation-wrong-gender-female-anti':'simple annotate',
    'ambiguous-translation-wrong-gender-female-pro':'simple annotate',
    'ambiguous-translation-wrong-gender-male-anti':'simple annotate',
    'ambiguous-translation-wrong-gender-male-pro':'simple annotate',
    'ambiguous-translation-wrong-sense-frequent':'diff_flexible',
    'ambiguous-translation-wrong-sense-infrequent':'diff_flexible',
    'anaphoric_group_it-they:deletion':'simple annotate',
    'anaphoric_group_it-they:substitution':'simple annotate',
    'anaphoric_intra_non-subject_it:deletion':'simple annotate',
    'anaphoric_intra_non-subject_it:substitution':'simple annotate',
    'anaphoric_intra_subject_it:deletion':'simple annotate',
    'anaphoric_intra_subject_it:substitution':'simple annotate',
    'anaphoric_intra_they:deletion':'simple annotate',
    'anaphoric_intra_they:substitution':'simple annotate',
    'anaphoric_singular_they:deletion':'simple annotate',
    'anaphoric_singular_they:substitution':'simple annotate',
    'antonym-replacement':'',
    'commonsense-only-ref-ambiguous':'',
    'commonsense-src-and-ref-ambiguous':'',
    'copy-source':'',
    'coreference-based-on-commonsense':'REF_flexible',
    'do-not-translate':'',
    'hallucination-date-time':'simple annotate',
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
    'hypernym-replacement':'',
    'hyponym-replacement':'',
    'lexical-overlap':'?',
    'modal_verb:deletion':'simple annotate',
    'modal_verb:substitution':'diff_flexible',
    'nonsense':'REF_flexible',
    'omission':'simple annotate',
    'ordering-mismatch':'swap',
    'overly-literal-vs-correct-idiom':'diff_flexible',
    'overly-literal-vs-explanation':'diff_flexible',
    'overly-literal-vs-ref-word':'diff_flexible',
    'overly-literal-vs-synonym':'diff_flexible',
    'pleonastic_it:deletion':'',
    'pleonastic_it:substitution':'',
    'punctuation:deletion_all':'',
    'punctuation:deletion_commas':'',
    'punctuation:deletion_quotes':'',
    'punctuation:statement-to-question':'',
    'real-world-knowledge-entailment':'',
    'real-world-knowledge-hypernym-vs-distractor':'',
    'real-world-knowledge-hypernym-vs-hyponym':'',
    'real-world-knowledge-synonym-vs-antonym':'',
    'similar-language-high':'',
    'similar-language-low':'',
    'untranslated-vs-ref-word':'',
    'untranslated-vs-synonym':'',
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
    
logger.setLevel(logging.INFO)
for idx,sample in tqdm(enumerate(dataset["train"])):
    if sample["phenomena"] in phenomena.keys() and phenomena[sample["phenomena"]] == 'diff_flexible':
        if phenomena[sample["phenomena"]] == 'simple annotate':
            logger.debug("in simple annotate")
            """
            # check if ambiguous-translation-wrong-sense-infrequent (?) has only same size replacements
            g = sample["good-translation"].split()
            b = sample["incorrect-translation"].split()
            if len(g) != len(b):
                print(idx)
                break
            """
            try:
                change = annotate(sample["good-translation"], sample["incorrect-translation"])
            except:
                logger.warning('error in id {}'.format(idx))
                break
            if len(change) == 0:
                logger.warning('No change in id {}'.format(idx))
            else:
                sample['annotation'] = change
                annotations[idx] = sample

        elif phenomena[sample["phenomena"]] == 'diff_flexible':
            g, g_spans = tokenize(sample["good-translation"])
            b, b_spans = tokenize(sample["incorrect-translation"])
            
            if len(g) == len(b):
                change = diff(g, g_spans, b, b_spans, phenomena="replacement")
            else:
                change = diff_flexible(sample["good-translation"], g, g_spans, sample["incorrect-translation"], b, b_spans)
            if len(change) == 0 and g != b:
                logger.warning('No change in id {}'.format(idx), g,b,change)
            else:
                sample['annotation'] = change
                annotations[idx] = sample

        elif phenomena[sample["phenomena"]] == 'REF_flexible':
            b, b_spans = tokenize(sample["incorrect-translation"])
            r, r_spans = tokenize(sample["reference"])
            
            if len(r) == len(b):
                change = diff(r, r_spans, b, b_spans, phenomena="replacement")
            else:
                change = diff_flexible(sample["reference"], r, r_spans, sample["incorrect-translation"], b, b_spans)
            if len(change) == 0 and g != b:
                logger.warning('No change in id {}, \nb: {},\nr: {}'.format(idx, b, r))
            else:
                sample['annotation'] = change
                annotations[idx] = sample
                
        elif phenomena[sample["phenomena"]] == 'units':
            try:
                g, b, change = annotate_units(sample["good-translation"],sample["incorrect-translation"])
            except: 
                print(idx)
                break
            if len(change) == 0 and g != b:
                logger.warning('No change in id {}, \ng: {}, \nb: {},\nr: {}'.format(idx, g, b, r))
            else:
                sample['annotation'] = change
                annotations[idx] = sample
                if len(change) > 1:
                    print(idx)
                    break
                
        elif phenomena[sample["phenomena"]] == 'swap':
            try:
                change = annotate_swap(sample["good-translation"],sample["incorrect-translation"])
            except: 
                print(idx)
                break
            sample['annotation'] = change
            annotations[idx] = sample

with open(annotated_dataset_path, "w") as f:
    json.dump(annotations, f)  # encode dict into JSON
logger.info("Done writing dict into {} file".format(annotated_dataset_path))