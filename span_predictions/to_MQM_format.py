#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, logging, csv
from tqdm import tqdm
import pandas as pd
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from format_utilities import annotation_to_MQM_sample
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "ACES_private/aces")))
from utils import read_file

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--data_path", help="path to annotated dataset, a tsv file with our annotation format or a folder (merged.tsv)")
    parser.add_argument("-o", "--out_path", help="path to the file to save the output data (csv file)")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        logger.error("The input path {} does not exists. See the challenge_set_annotation folder to download/annotate the dataset".format(args.data_path))
        exit()
    if os.path.exists(args.out_path):
        logger.warning("Overwriting {}".format(args.out_path))
        
    # ------------------------------------ READ ACES format data ---------------------------------------------------
    logger.info('{} is being loaded'.format(args.data_path))
    incorrect_samples = {}
    good_samples = {}
    content = read_file(args.data_path)
    content = content.reset_index()  # make sure indexes pair with number of rows
    for index, row in content.iterrows():
        # here, we add 2 samples: one where mt=incorrect-translation and one where mt:good-translation
        # in the good-translation samples, we don't annotate them so for the annotation, we can just give the good translation again
        idx = int(row["ID"])
        incorrect_samples[idx] = {
            'src':row['source'],
            'mt':row['incorrect-translation'],
            'ref':row['reference'],
            'score':None,
            'system':row["annotation-method"],
            'lp':row['langpair'],
            'segid':idx,
            'annotation':row['incorrect-translation-annotated'],
            'incorrect':True
        } 
        good_samples[idx] = {
            'src':row['source'],
            'mt':row['good-translation'],
            'ref':row['reference'],
            'score':None,
            'system':row["annotation-method"],
            'lp':row['langpair'],
            'segid':idx,
            'annotation':row['good-translation'],
            'incorrect':False
        }                            
    
    # Convert dict format to csv format
    logger.info('Converting...')
    
    cols = ["src", "mt", "ref", "score", "system", "lp", "segid", "annotation", "incorrect"]
    df_incorrect = pd.DataFrame.from_dict(incorrect_samples, orient='index', columns=cols)
    df_good = pd.DataFrame.from_dict(good_samples, orient='index', columns=cols)
    df = pd.concat([df_incorrect, df_good])
    df.to_csv(args.out_path, index=True, quoting=csv.QUOTE_NONE, escapechar="\\")
    logger.info('Saving to {}, length: {}'.format(args.out_path, len(df)))

    # reading this csv:
    # content = pd.read_csv("ACES_final_merged_MQM.csv", quoting=csv.QUOTE_NONE, escapechar="\\")