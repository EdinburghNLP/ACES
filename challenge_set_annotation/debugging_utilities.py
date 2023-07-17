#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os
from IPython.display import display, HTML
from datasets import load_from_disk

from annotation_utilities import *

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
    'coreference-based-on-commonsense':'mixed_flexible',
    'do-not-translate':'whole_sentence',
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
    'lexical-overlap':'manual',
    'modal_verb:deletion':'add-omit',
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
    'punctuation:deletion_all':'add-omit',
    'punctuation:deletion_commas':'add-omit',
    'punctuation:deletion_quotes':'add-omit',
    'punctuation:statement-to-question':'add-omit',
    'real-world-knowledge-entailment':'diff_flexible',
    'real-world-knowledge-hypernym-vs-distractor':'diff_flexible',
    'real-world-knowledge-hypernym-vs-hyponym':'diff_flexible',
    'real-world-knowledge-synonym-vs-antonym':'diff_flexible',
    'similar-language-high':'whole_sentence',
    'similar-language-low':'whole_sentence',
    'untranslated-vs-ref-word':'diff_flexible',   # here add-omit can be used for getting character level replacements too
    'untranslated-vs-synonym':'diff_flexible',
    'xnli-addition-contradiction':'manual',
    'xnli-addition-neutral':'manual',
    'xnli-omission-contradiction':'manual',
    'xnli-omission-neutral':'manual'
}

def display_annotation(idx, sample):
    if "annotation" not in sample:
        display_sample(idx, sample)
        return 
    m1 = "<mark>"
    m2 = "</mark>"    
    # good sentence: good-translation or reference?
    if phenomena[sample["phenomena"]] == "REF_flexible":
        good = sample["reference"]
    elif phenomena[sample["phenomena"]] == "mixed_flexible":
        good = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"])
    else:
        good = sample["good-translation"]
        
    change = sample["annotation"]
    bad = sample["incorrect-translation"]
    bad_new = ""
    
    # if the annotation is not standardized (it should be) we can still check if it is omission or not
    omission = True
    for c in change:
        if c["in_bad"] != None:
            omission = False
    left = 0
    for c in change:
        if "in_bad" in c and c["in_bad"] != None:
            span = c["in_bad"]["character_span"]
            bad_new += bad[left:span[0]] + m1 + bad[span[0]:span[1]] + m2
            left = span[1]
        elif "in_good" in c and c["in_good"] != None and omission:
            span = c["in_good"]["character_span"]
            bad_new += good[left:span[0]] + m1 + "(omission)" + m2
            left = span[1]
    if omission:
        bad_new += good[left:]
    else:
        bad_new += bad[left:]
    html = '''
    <style>
    mark {{
        background-color: #4CAF50;
        color: black;
      }}
    </style>
    <body>
    <h3>{}</h3>
    <p>Source: {}</p>
    <p>Reference: {}</p>
    <p>Good translation: {}</p>
    <p>Incorrect: {}</p>
    <p>Phenomenon: {}</p>
    </body>
    '''.format(idx, sample['source'], sample['reference'], sample['good-translation'], bad_new, sample['phenomena'])
    display(HTML(html))
   
def display_sample(idx, sample):
    html = '''
    <style>
    mark {{
        background-color: #4CAF50;
        color: black;
      }}
    </style>
    <body>
    <h3>{}</h3>
    <p>Source: {}</p>
    <p>Reference: {}</p>
    <p>Good translation: {}</p>
    <p>Incorrect: {}</p>
    <p>Phenomenon: {}</p>
    </body>
    '''.format(idx, sample['source'], sample['reference'], sample['good-translation'], sample['incorrect-translation'], sample['phenomena'])
    display(HTML(html))
    
# given a manually annotated sample (where there are <> in incorrect and good/reference sentences)
# calculate the character spans in the original sentences and return the change in our annotation format
def calculate_change(good, bad, sample):  
    bad_id = 0
    span = False # False is when we are not inside a span, True is inside a span
    change = []
    for i, c in enumerate(bad):
        if c == "<":
            if span:
                logger.error("< not closed. Try again.\n")
                return manual_annotation(".", sample)
            else:
                start = bad_id
                start_annotate = i
                bad_id -= 1
                span = True
        elif c == ">":
            if not span:
                logger.error("No opening < Try again.\n")
                return manual_annotation(".", sample)
            else:
                change.append({"in_good":None, 
                    "in_bad":{'token_index':None, 
                    'character_span':(start,bad_id), 
                               'token':bad[start_annotate+1:i]}})
                bad_id -= 1
                span = False
        bad_id += 1
    good_id = 0
    span = False # False is when we are not inside a span, True is inside a span
    for i, c in enumerate(good):
        if c == "<":
            if span:
                logger.error("< not closed. Try again.\n")
                return manual_annotation(".", sample)
            else:
                start = good_id
                start_annotate = i
                good_id -= 1
                span = True
        elif c == ">":
            if not span:
                logger.error("No opening < Try again.\n")
                return manual_annotation(".", sample)
            else:
                change.append({"in_good":{'token_index':None, 
                    'character_span':(start,good_id), 
                               'token':good[start_annotate+1:i]}, 
                    "in_bad":None})
                good_id -= 1
                span = False
        good_id += 1
    return change
  
# the UI (?) part of the manual annotation
def manual_annotation(idx, sample, inp="."):
    while inp != "":
        inp = input("Enter the incorrect translation with the < and > to show the error spans (exit to stop, skip to skip): \n")
        bad = inp
        if bad == "exit":
            return -1
        if bad == "skip":
            return 1
        inp = input("Enter the correct/reference translation with the < and > to show the error spans (exit to stop, skip to skip): \n")
        good = inp
        if good == "exit":
            return -1
        if good == "skip":
            return 1
        
        change = calculate_change(good, bad, sample)
        tmp = copy.deepcopy(sample)
        tmp["annotation"] = change
        display_annotation(idx, tmp)
        inp = input("\n To accept it press enter or to annotate again enter any other string: ")
        if inp == "":
            sample['annotation'] = change
            sample['method'] = "manual annotation"
            return sample
    return 2    # will never reach here

# the UI (?) part of the annotation in general (ask if they want to accept the annotation, call manual_annotation if no)
def manual_annotation_io_warnings(idx, sample):
    if phenomena[sample["phenomena"]] in ['?', 'mixed_flexible']:
        print("-----> For this sample we can compare the Incorrect translation with either Reference or Good translation.")
    elif phenomena[sample["phenomena"]] in ['REF_flexible']:
        print("-----> For this sample we compare the Incorrect translation with the Reference.")
    else:
        print("-----> For this sample we compare the Incorrect translation with the Good translation.\n")
    display_annotation(idx, sample)
    inp = input('To accept the suggested annotation click on enter. To skip this one enter skip. To add to the manually annotated files enter add. To exit enter exit and enter anything else to manually annotate:')
    if inp == "skip":
        return 1
    if inp == "exit":
        return -1
    if inp == "add":
        return 2
    res = manual_annotation(sample, inp)
    return res
        
def is_word_level(g_spans, b_spans, change):
    g_starts = [span[0] for span in g_spans]
    b_starts = [span[0] for span in b_spans]
    g_ends = [span[1] for span in g_spans]
    b_ends = [span[1] for span in b_spans]
    for c in change:
        if "in_good" in c and c["in_good"] != None:
            if c["in_good"]["character_span"][0] not in g_starts or c["in_good"]["character_span"][1] not in g_ends:
                return False
        if "in_bad" in c and c["in_bad"] != None:
            if c["in_bad"]["character_span"][0] not in b_starts or c["in_bad"]["character_span"][1] not in b_ends:
                return False
    return True

def check_warnings(stats, phenomena_tobe_processed, annotations):
    for problem in ["too_long", "other"]:
        for (idx, _) in stats[phenomena_tobe_processed][problem]:
            # if the function returns 0 it automatically accepted, if 1 skipping and if -1 exit
            if idx in annotations:
                sample = annotations[idx]
            else:
                sample = dataset["train"][idx]
            res = manual_annotation_io_warnings(idx, sample) 
            # if exit, first save a new annotations file to save progress and then exit
            if res == -1:
                return -1
            
folder = os.getcwd()
dataset_path = os.path.join(folder, 'dataset')
if not os.path.exists(dataset_path):
    logger.error('No dataset path in debugging_utilities.py: %s' %(dataset_path))
dataset = load_from_disk(dataset_path)