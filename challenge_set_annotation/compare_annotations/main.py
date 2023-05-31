#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import numpy as np
import json

class Annotations(object):
    def __init__(self, test):
        self.files = []
        with open(test, "r") as f:
            self.raw = json.load(f)
        self.i = -1
        self.keys = list(self.raw.keys())
        self.files = [{}, {}, {}, {}, {}]
    
    def add(self, data):
        print("in add ")
        for i,f in enumerate(self.files):
            if f == {}:
                # change all to string
                annotations = json.loads(data)
                annotations = {int(k):v for k,v in annotations.items()}
                self.files[i] = annotations
                print("add succesful")
                return 
        
    def set_i(self, i):
        self.i = i
        
    def get_item(self):
        res = ["-", "-", "-", "-", "-"]
        if self.i >= len(self.keys):
            print("finished")
            return -1
        if self.i < 0:
            print("No more back")
            return -1
        key = self.keys[self.i]
        sample = self.raw[key]
        raw_text = "Reference: {}<br />Good translation: {}<br>Incorrect translation: {}<br />Phenomenon: {}".format(sample["reference"], sample["good-translation"], sample["incorrect-translation"], sample["phenomena"])
        res[0] = raw_text
        print(raw_text)
        for i,f in enumerate(self.files):
            if f != {} and key in f:
                sample = f[key]
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
        