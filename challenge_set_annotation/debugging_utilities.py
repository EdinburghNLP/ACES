#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os
from IPython.display import display, HTML
from datasets import load_from_disk

from annotation_utilities import *

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
  
# the low level part of manual annotation: keep asking for an annotation until one is accepted
def manual_annotation(idx, sample, inp="."):
    while inp != '':
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
            omission = True
            for c in change:
                if c["in_bad"] != None:
                    omission = False
            sample["omission"] = omission
            return sample
    return 

# the UI (?) part of the annotation in general (ask if they want to accept the annotation, call manual_annotation if no)
def manual_annotation_io(idx, sample):
    if phenomena[sample["phenomena"]] in ['?', 'mixed_flexible']:
        print("-----> For this sample we can compare the Incorrect translation with either Reference or Good translation.")
    elif phenomena[sample["phenomena"]] in ['REF_flexible']:
        print("-----> For this sample we compare the Incorrect translation with the Reference.")
    else:
        print("-----> For this sample we compare the Incorrect translation with the Good translation.\n")
    
    display_annotation(idx, sample)
    if 'annotation' in sample:
        inp = input('To accept the suggested annotation click on enter. To skip this one enter skip. To add to the manually annotated files enter add. To exit enter exit and enter anything else to manually annotate:')
    else:
        inp = input('To skip this one enter skip. To add to the manually annotated files enter add. To exit enter exit and enter anything else to manually annotate:')
    if inp == "skip":
        return 1
    if inp == "exit":
        return -1
    if inp == "add":
        return 2
    if inp == "":
        return sample
    res = manual_annotation(idx, sample, inp)
    assert res in [1, -1] or 'annotation' in res
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
          
folder = os.getcwd()
dataset_path = os.path.join(folder, 'dataset')
if not os.path.exists(dataset_path):
    logger.error('No dataset path in debugging_utilities.py: %s' %(dataset_path))
dataset = load_from_disk(dataset_path)