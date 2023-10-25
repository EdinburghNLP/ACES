import os, sys, csv, re
sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "ACES_private/challenge_set_annotation")))
from annotation_utilities import *
from debugging_utilities import *

"""_summary_
Helper functions to convert between these 3 formats:
1. Json format (saved in .txt):
    [{'in_good': {'token_index': […],
        'character_span': (117, 159),
        'token': 'adjective is used in a substantivized form'},
    'in_bad': {'token_index': […],
        'character_span': (117, 156),
        'token': 'attribute word is used as the main word'}}]
    - Read this format as a dictionary
    
2. TSV format:
    Columns: ID     TYPE	            A	                    B
    Row:    1023    hallucination       good-translation        incorrect-translation (annotated or not)
    - Read this format as pd DataFrame
    
3. MQM (CSV) format:
    Columns:    src,    mt,                     ref,        score,  system, lp,     segid,  annotation
    Row:        source  incorrect-translation   reference   None    ACES    en-de   ID      annotated incorrect-translation
    - Read this format as pd DataFrame
    
"""

# read TSV and CSV in this format: [[ID, Type, A, B], ...]
def read_to_list(content):
    tsv_data = []
    for line in content.split('\n'):
        tsv_data.append(line.split('\t'))
    tsv_data = tsv_data[1:]
    return tsv_data

"""
# input format:
   translation-direction	source	good-translation	incorrect-translation	reference	phenomena
#       en-fr                  ""       ""                  ""                      ""       "addition"
# output format
      {'source': "Proper nutritional practices alone cannot generate elite performances, but they can significantly affect athletes' overall wellness.",
 'good-translation': 'Las prácticas nutricionales adecuadas por sí solas no pueden generar rendimiento de élite, pero pueden afectar significativamente el bienestar general de los atletas.',
 'incorrect-translation': 'Las prácticas nutricionales adecuadas por sí solas no pueden generar rendimiento de élite, pero pueden afectar significativamente el bienestar general de los jóvenes atletas.',
 'reference': 'No es posible que las prácticas nutricionales adecuadas, por sí solas, generen un rendimiento de elite, pero puede influir en gran medida el bienestar general de los atletas .',
 'phenomena': 'addition',
 'langpair': 'en-es'}
 """
 
def load_tsv_dataset(content):
    dataset = []
    for line in content.split('\n')[1:]:
        tsv_data = line.split('\t')
        if len(tsv_data) >= 6:
            dataset.append({
                "source":tsv_data[1],
                "good-translation":tsv_data[2],
                "incorrect-translation":tsv_data[3],
                "reference":tsv_data[4],
                "phenomena":tsv_data[5],
                "langpair":tsv_data[0]
            })
    return dataset

def save_tsv(tsv_data, path):
    with open(path, "wt") as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['ID', 'TYPE', 'A', 'B'])    
        for sample in tsv_data:
            tsv_writer.writerow(sample)    

def tsv_to_annotation(tsv_data):
    annotations = {}
    for sample in tsv_data:
        if len(sample) == 4:
            s = tsv_annotation_to_change(sample)
            annotations[int(sample[0])] = s
    return annotations

def annotation_to_tsv(annotations, ids=None):
    tsv_data = []
    if ids != None:
        for idx in ids:
            try:
                tsv_data.append(annotation_to_tsv_sample(idx, annotations[idx]))
            except:
                try:
                    process_sample(idx, samples[idx], manual=False, detokenize=True)
                    tsv_data.append(annotation_to_tsv_sample(idx, annotations[idx]))
                except:
                    print(idx)
    else:
        for idx, sample in annotations.items():
            try:
                tsv_data.append(annotation_to_tsv_sample(idx, annotations[idx]))
            except:
                try:
                    process_sample(idx, samples[idx], manual=False, detokenize=True)
                    tsv_data.append(annotation_to_tsv_sample(idx, annotations[idx]))
                except:
                    print(idx)
    return tsv_data

def annotation_to_tsv_sample(idx, sample):  
    if phenomena[sample["phenomena"]] == "REF_flexible":
        good = sample["reference"]
    elif phenomena[sample["phenomena"]] == "mixed_flexible":
        good = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"])
    else:
        good = sample["good-translation"]
    return [idx, sample["phenomena"], good, change_to_tsv_annotation(sample)]

# src,    mt,                     ref,        score,  system (phenomena), lp,     segid,  annotation, manual(True or False)
# what about we give error type to system? maybe we get some statistics
def annotation_to_MQM_sample(idx, sample, manual=False):  
    return [sample['source'], sample['incorrect-translation'], sample['reference'], None, sample['phenomena'], sample['langpair'], idx, change_to_tsv_annotation(sample, m1="<v>", m2="</v>"), manual]
    
# to change our annotation to MQM format: m1="<v>", m2="</v>"    
def change_to_tsv_annotation(sample, m1="<", m2=">"):
    # check ref or good too
    if phenomena[sample["phenomena"]] == "REF_flexible":
        good = sample["reference"]
    elif phenomena[sample["phenomena"]] == "mixed_flexible":
        good = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"])
    else:
        good = sample["good-translation"]

    bad = sample["incorrect-translation"]
    change = sample["annotation"]
    if "omission" in sample:
        omission = sample["omission"]
    else:
        omission = True
        for c in change:
            if c != None:
                omission = False
    left = 0
    bad_2 = ""
    try:
        for c in change:
            if c["in_bad"] != None:
                span = c["in_bad"]["character_span"]
                bad_2 += bad[left:span[0]] + m1 + bad[span[0]:span[1]] + m2
                left = span[1]
            elif c["in_good"] != None and omission:
                span = c["in_good"]["character_span"]
                bad_2 += good[left:span[0]] + m1 + m2
                left = span[1]
    except:
        print(sample)
    if omission:
        bad_2 += good[left:]
    else:
        bad_2 += bad[left:]
    return bad_2

def tsv_annotation_to_change(sample): 
    p = sample[1]
    good = sample[2]
    bad = sample[3]
    
    # check ref or good
    if p not in phenomena:
        ref_or_good = "good-translation"
    else:
        if phenomena[p] == "REF_flexible":
            ref_or_good = "reference"
        # we ignore this condition rn because we don't have access to the reference/good translation
        # elif phenomena[p] == "mixed_flexible":
        #    g_chosen = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"])
        else:
            ref_or_good = "good-translation"
    
    bad_id = 0
    span = False # False is when we are not inside a span, True is inside a span
    change = []
    # check if omission
    omission = False
    if "<>" in bad:
        omission = True
    for i, c in enumerate(bad):
        if c == "<":
            if span:
                sample["incorrect-translation"] = "There was an error: "+bad
                sample["annotation"] = []
                return sample
            else:
                start = bad_id
                start_annotate = i
                bad_id -= 1
                span = True
        elif c == ">":
            if not span:
                sample["incorrect-translation"] = "There was an error: "+bad
                sample["annotation"] = []
                return sample
            else:
                change.append({"in_good":None, 
                    "in_bad":{'token_index':None, 
                    'character_span':(start,bad_id), 
                               'token':bad[start_annotate+1:i]}})
                bad_id -= 1
                span = False
        bad_id += 1
        
    incorrect = "".join([c for c in bad if c not in ["<", ">"]])
    return {"phenomena":p, ref_or_good:good, "incorrect-translation":incorrect, "annotation":change, "omission":omission}

def tsv_annotation_to_MQM_annotation(bad, m1="<v>", m2="</v>"):
    bad_0 = re.sub(r"<", "{", bad)
    bad_0 = re.sub(r">", "}", bad_0)
    bad_0 = re.sub(r"{", m1, bad_0)
    bad_0 = re.sub(r"}", m2, bad_0)
    return bad_0