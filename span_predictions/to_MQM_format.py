#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, json, logging
from tqdm import tqdm
import pandas as pd
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from format_utilities import annotation_to_MQM_sample

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-d", "--data_path", help="path to annotated dataset, a txt/json file with our annotation format or a folder")
    parser.add_argument("-o", "--out_path", help="path to the file to save the output data")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        logger.error("The input path {} does not exists. See the challenge_set_annotation folder to download/annotate the dataset".format(args.data_path))
        exit()
    if os.path.exists(args.out_path):
        logger.warning("Overwriting {}".format(args.out_path))
        
    logger.info('{} is being loaded'.format(args.data_path))
    with open(args.data_path, "r") as f:
        annotations = json.load(f)
    annotations = {int(k):v for k,v in annotations.items()}
    
    logger.info('Converting...')
    rows = [annotation_to_MQM_sample(idx, sample) for (idx,sample) in tqdm(annotations.items())]

    # creating df object with columns specified    
    cols = ["src", "mt", "ref", "score", "system", "lp", "segid", "annotation", "manual"]
    df = pd.DataFrame(rows, columns =cols)
    
    logger.info('Saving to {}'.format(args.out_path))
    df.to_csv(args.out_path, index=False)