#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numpy as np
import json
from annotation_utilities import *

# given a manually annotated sample (where there are <> in incorrect and good/reference sentences)
# calculate the character spans in the original sentences and return the change in our annotation format
def calculate_change(sample): 
    p = sample[1]
    good = sample[2]
    bad = sample[3]
    bad_id = 0
    span = False # False is when we are not inside a span, True is inside a span
    change = []
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
    return {"phenomena":p, "good-translation":good, "incorrect-translation":incorrect, "annotation":change}

def change_to_annotation(sample, m1, m2):
    print("in change to annotate")
    if "bad_2" in sample:
        print(sample)
        return sample["bad_2"]
    
    # check ref or good too
    if "reference" in sample:
        if phenomena[sample["phenomena"]] == "REF_flexible":
            good = sample["reference"]
        elif phenomena[sample["phenomena"]] == "mixed_flexible":
            good = ref_or_good(sample["reference"], sample["good-translation"], sample["incorrect-translation"])
        else:
            good = sample["good-translation"]
    else:
        good = sample["good-translation"]
    bad = sample["incorrect-translation"]
    print("good: ", good, " bad: ", bad)
    change = sample["annotation"]
    left = 0
    bad_2 = ""
    omission = True
    for c in change:
        if c["in_bad"] != None:
            omission = False
            span = c["in_bad"]["character_span"]
            bad_2 += bad[left:span[0]] + m1 + bad[span[0]:span[1]] + m2
            left = span[1]
        elif c["in_good"] != None and omission:
            span = c["in_good"]["character_span"]
            bad_2 += good[left:span[0]] + m1 + "(omission)" + m2
            print("bad2 in omission: ", bad_2)
            left = span[1]
    if omission:
        bad_2 += good[left:]
    else:
        bad_2 += bad[left:]
    print("bad2: ", bad_2)
    return bad_2

def process_tsv(content, m1=None, m2=None):
    lines = []
    for line in content.split('\n'):
        lines.append(line.split('\t'))
    lines = lines[1:]
    
    annotations = {}
    for sample in lines:
        if len(sample) == 4:
            s = calculate_change(sample)
            if m1 != None:
                bad_2 = sample[3]
                omission = m1+m2
                space = m1+"(omission)"+m2
                bad_2_0_1 = re.sub(r"<", "{", bad_2)
                bad_2_0_2 = re.sub(r">", "}", bad_2_0_1)
                bad_2_2 = re.sub(r"{", m1, bad_2_0_2)
                bad_2_3 = re.sub(r"}", m2, bad_2_2)
                bad_2_4 = re.sub(re.escape(omission), space, bad_2_3)
                
                s["bad_2"] = bad_2_4
            annotations[sample[0]] = s
    
    return annotations

class Annotations(object):
    def __init__(self, n_files):
        self.n_files = n_files
        self.files = [{} for _ in range(n_files)]
    
    def load(self, data, tsv=False):
        if tsv:
            self.raw = process_tsv(data)
        else:
            annotations = json.loads(data)
            self.raw = {str(k):v for k,v in annotations.items()}
        self.i = -1
        # self.keys = list(self.raw.keys())
        self.keys = [17418, 10007, 14685, 33876, 8614, 13482, 34747, 4172, 32266, 13469, 33409, 33457, 15685, 31415]
            
    def add(self, data, tsv=False):
        print("in add ")
        for i,f in enumerate(self.files):
            if f == {}:
                # some cleaning - if there are string keys, then they are probably more recent.
                # so if there are duplicate keys (id, 'id'), then when we convert everything to string,
                # the more recent one will be saved - which should be fine.
                if tsv:
                    m1 = "<mark"+str(i+1)+">"
                    m2 = "</mark"+str(i+1)+">"
                    self.files[i] = process_tsv(data, m1, m2)
                else:
                    annotations = json.loads(data)
                    self.files[i] = {str(k):v for k,v in annotations.items()}
                print("add succesful")
                return 
        
    def set_i(self, i):
        self.i = i
        
    def get_item(self):
        print("in get_item")
        res = ["-" for _ in range(self.n_files+1)]
        if self.i >= len(self.keys):
            print("finished")
            return -1
        if self.i < 0:
            print("No more back")
            return -1
        try:
            key = str(self.keys[self.i])
        except:
            print("wtf? ", self.i)
            print(self.keys)
        try:
            sample = self.raw[key]
        except:
            print("key not found, ", key, self.raw.keys())

        if "reference" in sample:
            raw_text = "ID: {}<br />Reference: {}<br />Good translation: {}<br>Incorrect translation: {}<br />Phenomenon: {}".format(key, sample["reference"], sample["good-translation"], sample["incorrect-translation"], sample["phenomena"])
        else:
            raw_text = "ID: {}<br />A: {}<br>B: {}<br />Phenomenon: {}".format(key, sample["good-translation"], sample["incorrect-translation"], sample["phenomena"])            
        res[0] = raw_text
        for i,f in enumerate(self.files):
            if f != {} and key in f:
                sample = f[key]
                # annotations = sample["annotation"]
                # bad = sample["incorrect-translation"]
                # bad_new = bad
                # n_spans = 0
                m1 = "<mark"+str(i+1)+">"
                m2 = "</mark"+str(i+1)+">"
                """
                m_len = len(m1) + len(m2)
                for annotation in annotations:
                    if annotation["in_bad"] != None:
                        span = annotation["in_bad"]["character_span"]
                        indices = (span[0]+n_spans*m_len, span[1]+n_spans*m_len)
                        bad_new = bad_new[:indices[0]] + m1 + bad_new[indices[0]:indices[1]] + m2 + bad_new[indices[1]:]
                        n_spans += 1
                """
                bad_new = change_to_annotation(sample, m1, m2)
                res[i+1] = bad_new
                
        return res
    
    def next(self):
        self.i += 1
        return self.get_item()

    def back(self):
        self.i -= 1
        return self.get_item()
        