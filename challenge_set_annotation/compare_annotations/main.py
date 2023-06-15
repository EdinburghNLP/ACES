#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numpy as np
import json

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

def process_tsv(content):
    lines = []
    for line in content.split('\n'):
        lines.append(line.split('\t'))
    lines = lines[1:]
    annotations = {}
    for sample in lines:
        if len(sample) == 4:
            annotations[sample[0]] = calculate_change(sample)
    return annotations

class Annotations(object):
    def __init__(self):
        self.files = [{}, {}, {}, {}, {}]
    
    def load(self, data, tsv=False):
        if tsv:
            self.raw = process_tsv(data)
        else:
            annotations = json.loads(data)
            self.raw = {str(k):v for k,v in annotations.items()}
        self.i = -1
        self.keys = list(self.raw.keys())
            
    def add(self, data, tsv=False):
        print("in add ")
        for i,f in enumerate(self.files):
            if f == {}:
                # some cleaning - if there are string keys, then they are probably more recent.
                # so if there are duplicate keys (id, 'id'), then when we convert everything to string,
                # the more recent one will be saved - which should be fine.
                if tsv:
                    self.files[i] = process_tsv(data)
                else:
                    annotations = json.loads(data)
                    self.files[i] = {str(k):v for k,v in annotations.items()}
                print("add succesful")
                return 
        
    def set_i(self, i):
        self.i = i
        
    def get_item(self):
        print("in get_item")
        res = ["-", "-", "-", "-", "-"]
        if self.i >= len(self.keys):
            print("finished")
            return -1
        if self.i < 0:
            print("No more back")
            return -1
        key = self.keys[self.i]
        sample = self.raw[key]
        if "reference" in sample:
            raw_text = "Reference: {}<br />Good translation: {}<br>Incorrect translation: {}<br />Phenomenon: {}".format(sample["reference"], sample["good-translation"], sample["incorrect-translation"], sample["phenomena"])
        else:
            raw_text = "A: {}<br>B: {}<br />Phenomenon: {}".format(sample["good-translation"], sample["incorrect-translation"], sample["phenomena"])            
        res[0] = raw_text
        for i,f in enumerate(self.files):
            if f != {} and key in f:
                sample = f[key]
                print(sample["annotation"])
                annotations = sample["annotation"]
                bad = sample["incorrect-translation"]
                bad_new = bad
                n_spans = 0
                m1 = "<mark"+str(i+1)+">"
                m2 = "</mark"+str(i+1)+">"
                m_len = len(m1) + len(m2)
                for annotation in annotations:
                    if annotation["in_bad"] != None:
                        span = annotation["in_bad"]["character_span"]
                        indices = (span[0]+n_spans*m_len, span[1]+n_spans*m_len)
                        bad_new = bad_new[:indices[0]] + m1 + bad_new[indices[0]:indices[1]] + m2 + bad_new[indices[1]:]
                        n_spans += 1
                res[i+1] = bad_new
        return res
    
    def next(self):
        self.i += 1
        return self.get_item()

    def back(self):
        self.i -= 1
        return self.get_item()
        