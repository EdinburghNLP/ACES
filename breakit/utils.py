#!/usr/bin/env python3

import pandas as pd
import csv


def read_file(filename: str) -> pd.DataFrame:
    '''
    Read TSV file and return as pandas DataFrame
    '''
    return pd.read_csv(filename, sep='\t', quoting=csv.QUOTE_NONE)
