#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, logging, json, glob, csv
from tqdm import tqdm
from datasets import load_from_disk
import pandas as pd
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "ACES_private/span_predictions")))
from format_utilities import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "ACES_private/challenge_set_annotation")))
from annotation_utilities import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "ACES_private/aces")))
from utils import read_file

"""
Given these optional paths:
1. path to the folder containing manual annotations
2. Path to the folder containing second round of manual annotations
3. Automatic annotations (in json format)
Merge everything and save to the specified output file (in MQM format)
"""

def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        logger.error('No dataset path: %s. See ACES_private/challenge_set_annotation/download_dataset.py to download it from HuggingFace' %(dataset_path))
        exit()
    logger.info('Loading the dataset...')
    dataset = load_from_disk(dataset_path)
    logger.info('Dataset loaded.')
    return dataset['train']

# MQM columns: src,    mt,                     ref,        score,  system, lp,     segid,  annotation,  manual
if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    
    # optional
    parser.add_argument("-a", "--automatic", help="path to automatically annotated dataset, a txt/json file with our annotation format or a folder - or tsv file")
    parser.add_argument("-m1", "--manual_1", help="path to manually annotated dataset, a folder with tsv files")
    parser.add_argument("-m2", "--manual_2", help="path to the second round of annotated dataset")
    
    # required
    parser.add_argument("-o", "--out_path", required=str, help="path to the file to save the output data")
    parser.add_argument("-d", "--dataset", required=str, help="path to the folder where ACES dataset is downloaded.")
    args = parser.parse_args()
    
    if os.path.exists(args.out_path):
        logger.warning("Overwriting {}".format(args.out_path))
    
    # 1. load the full dataset
    dataset = load_dataset(args.dataset)
    print('Full dataset size: ', len(dataset))
    
    # using dictionary because that won't allow duplicates
    samples = {}
    
    # 2. if there is a path to the automatic annotations load them and convert to the MQM format
    if args.automatic:
        logger.info('{} is being loaded'.format(args.automatic))
        annotations = read_json(args.automatic)
        annotations = {int(k):v for k,v in annotations.items()}
        logger.info('Converting...')
        for (idx,sample) in tqdm(annotations.items()):
            samples[idx] = annotation_to_ACES_sample(idx, sample)
        logger.info('No of automatically annotated samples: {}'.format(len(samples.keys())))
    # 3. if there is manual annotations read those in tsv format then convert to MQM
    m_ids = set()
    for p in [args.manual_1, args.manual_2]:
        if p:
            logger.info('{} is being loaded'.format(p))
            for file in glob.glob(os.path.join(p, "*.tsv")):
                content = read_file(file)
                content = content.reset_index()  # make sure indexes pair with number of rows
                for index, row in content.iterrows():
                    idx = int(row["ID"])
                    sample = dataset[idx]
                    samples[idx] = {
                        'source':sample['source'],
                        'good-translation':sample['good-translation'],
                        'incorrect-translation':sample['incorrect-translation'],
                        'reference':sample['reference'],
                        'phenomena':sample['phenomena'],
                        'langpair':sample['langpair'],
                        'incorrect-translation-annotated':tsv_annotation_to_MQM_annotation(row["B"]),
                        "annotation-method":"manual"
                    }                           
    logger.info('No of manually annotated samples: {}'.format(len(m_ids)))
    
    # -------------------------------------------------------------
    from collections import defaultdict
    
    manual_counts = defaultdict(int)
    for sample in samples.values():
        if sample["annotation-method"] == "manual":
            manual_counts[sample["phenomena"]] += 1
    print("manual_counts: ", manual_counts)
    logger.info('In total: {}'.format(len(samples.keys())))
    
    # 4. Create a df object with columns specified and save to CSV 
    df = pd.DataFrame.from_dict(samples, orient='index', columns=['source', 'good-translation', 'incorrect-translation', 'reference',  'phenomena', 'langpair', 'incorrect-translation-annotated', 'annotation-method'])
    df.index.name = 'ID'
    
    logger.info('Saving to {}'.format(args.out_path))
    df.to_csv(args.out_path, sep='\t', index=True, quoting=csv.QUOTE_NONE)
