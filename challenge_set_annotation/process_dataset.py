import datasets
from datasets import load_from_disk

import numpy as np
np.random.seed(42)
import random
import argparse, os, json
from tqdm import tqdm

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

# Span annotations for the addition data - word ids for now
# can handle multiple spans
def diff(g, b, phenomena="addition"):
    i, j = 0, 0
    change = []
    while j < len(b) and i < len(g):
        logger.debug([g[i], i, b[j], j])
        if g[i] == b[j]:
            i += 1
            j += 1
        else:
            if phenomena == "addition":
                change.append((j, b[j]))
                j += 1 
            elif phenomena == "replacement":
                change.append((i, g[i], b[j]))
                i += 1
                j += 1
            # for omission also return where the deleted part was supposed to be in incorrect translation
            elif phenomena == "omission":
                change.append((i, g[i], j))
                i += 1
    if phenomena == "addition":
        change.extend([(jx, b[jx]) for jx in range(j,len(b))])
    elif phenomena == "omission":
        change.extend([(ix, g[ix]) for ix in range(i,len(g))])
    return (g, b, change)

# this finds if any number of words are replaced with any numbe of words - but only for one span
# if there is more than one span, for example as in
# good = "What does slip ring starter and a b starter mean"
# bad = "What does savory starter and c starter mean"
# then it finds it as one big span: ('slip ring starter and a b', [2, 3, 4, 5, 6, 7], 'savory starter and c', [2, 3, 4, 5])
def diff_flexible(g, b, phenomena="word_sense"):
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
        change.append((" ".join(g[start:i+1]), list(range(start,i+1)), " ".join(b[start:j+1]), list(range(start,j+1))))
    # elif phenomena == "deletion":
    return g, b, change

def annotate(good, incorrect):
    g = good.split()
    b = incorrect.split()
    try:
        if len(g) > len(b):    # omission
            # return diff(b, g)
            return diff(g, b, phenomena="omission")
        
        elif len(g) < len (b):   # addition
            return diff(g, b)

        else:   # replacement
            return diff(g, b, phenomena="replacement")
    
    except:
        logging.error("Error in annotate!")
        return (g, b, [])

folder = os.getcwd()
dataset_path = os.path.join(folder, 'dataset')
if not os.path.exists(dataset_path):
    logger.error('No dataset path: '.format(dataset_path))
    exit()

logger.info('Loading the dataset...')
dataset = load_from_disk(dataset_path)
logger.info('Dataset loaded.')

# this is the list of phenomena and which option they need to be annotated with:
phenomena = {
    'addition':'simple annotate',
    'ambiguous-translation-wrong-discourse-connective-since-causal':'',
    'ambiguous-translation-wrong-discourse-connective-since-temporal':'',
    'ambiguous-translation-wrong-discourse-connective-while-contrast':'',
    'ambiguous-translation-wrong-discourse-connective-while-temporal':'',
    'ambiguous-translation-wrong-gender-female-anti':'',
    'ambiguous-translation-wrong-gender-female-pro':'',
    'ambiguous-translation-wrong-gender-male-anti':'',
    'ambiguous-translation-wrong-gender-male-pro':'',
    'ambiguous-translation-wrong-sense-frequent':'',
    'ambiguous-translation-wrong-sense-infrequent':'',
    'anaphoric_group_it-they:deletion':'',
    'anaphoric_group_it-they:substitution':'',
    'anaphoric_intra_non-subject_it:deletion':'',
    'anaphoric_intra_non-subject_it:substitution':'',
    'anaphoric_intra_subject_it:deletion':'',
    'anaphoric_intra_subject_it:substitution':'',
    'anaphoric_intra_they:deletion':'',
    'anaphoric_intra_they:substitution':'',
    'anaphoric_singular_they:deletion':'',
    'anaphoric_singular_they:substitution':'',
    'antonym-replacement':'',
    'commonsense-only-ref-ambiguous':'',
    'commonsense-src-and-ref-ambiguous':'',
    'copy-source':'',
    'coreference-based-on-commonsense':'',
    'do-not-translate':'',
    'hallucination-date-time':'',
    'hallucination-named-entity-level-1':'',
    'hallucination-named-entity-level-2':'',
    'hallucination-named-entity-level-3':'',
    'hallucination-number-level-1':'',
    'hallucination-number-level-2':'',
    'hallucination-number-level-3':'',
    'hallucination-real-data-vs-ref-word':'',
    'hallucination-real-data-vs-synonym':'',
    'hallucination-unit-conversion-amount-matches-ref':'',
    'hallucination-unit-conversion-unit-matches-ref':'',
    'hypernym-replacement':'',
    'hyponym-replacement':'',
    'lexical-overlap':'',
    'modal_verb:deletion':'',
    'modal_verb:substitution':'',
    'nonsense':'',
    'omission':'simple annotate',
    'ordering-mismatch':'',
    'overly-literal-vs-correct-idiom':'',
    'overly-literal-vs-explanation':'',
    'overly-literal-vs-ref-word':'',
    'overly-literal-vs-synonym':'',
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
    'xnli-addition-contradiction':'',
    'xnli-addition-neutral':'',
    'xnli-omission-contradiction':'',
    'xnli-omission-neutral':''
}

# if there are already some annotations overwrite them and append new ones
annotated_dataset_path = os.path.join(folder, 'annotated.txt')
if os.path.exists(annotated_dataset_path):
    logger.info('Path {} already exists. Loading..'.format(annotated_dataset_path))
    with open(annotated_dataset_path, "r") as f:
        annotations = json.load(f)
else:
    annotations = dict()
    
for idx,sample in tqdm(enumerate(dataset["train"])):
    if phenomena[sample["phenomena"]] == 'simple annotate':
        """
        # check if ambiguous-translation-wrong-sense-infrequent (?) has only same size replacements
        g = sample["good-translation"].split()
        b = sample["incorrect-translation"].split()
        if len(g) != len(b):
            print(idx)
            break
        """
        g, b, change = annotate(sample["good-translation"], sample["incorrect-translation"])
        if len(change) == 0:
            logger.warning('No change in id {}'.format(idx), g,b,change)
        else:
            sample['annotation'] = change
            annotations[idx] = sample

    elif sample["phenomena"] in []:
        g = sample["good-translation"].split()
        b = sample["incorrect-translation"].split()

        # check if it has multiple scans
        g, b, change = diff_flexible(g,b)
        if len(change) == 0 and g != b:
            logger.warning('No change in id {}'.format(idx), g,b,change)
        else:
            sample['annotation'] = change
            annotations[idx] = sample

    elif sample["phenomena"] in ["hallucination-named-entity-level-2"]:
        g = sample["reference"].split()
        b = sample["incorrect-translation"].split()

        # check if it has multiple scans
        g, b, change = diff_flexible(g,b)
        if len(change) == 0 and g != b:
            logger.warning('No change in id {}'.format(idx), g,b,change)
        else:
            sample['annotation'] = change
            annotations[idx] = sample

with open(annotated_dataset_path, "w") as f:
    json.dump(annotations, f)  # encode dict into JSON
logger.info("Done writing dict into {} file".format(annotated_dataset_path))