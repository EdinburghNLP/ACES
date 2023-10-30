"""
            python ACES_private/span_predictions/save_span_predictions.py -m {MODEL_PATH} -d merged.tsv --threshold 0.03
            --good span_baseline/baseline1_comet/ACES_alignment_scores_good/ 
            --bad span_baseline/baseline1_comet/ACES_alignment_scores_incorrect/
            -o span_baseline/baseline1_comet/ACES_predicted_spans_

"""
from comet import download_model, load_from_checkpoint # conda install?

import argparse, os, sys, json, logging, csv
from tqdm import tqdm
import pandas as pd
import numpy as np
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from scripts.predict_evaluate_spans_utils import *
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "ACES_private/aces")))
from utils import read_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save predicted spans"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model or Path to a checkpoint file.",
        type=str,
    )
    parser.add_argument(
        "-d", "--dataset",
        required=True,
        help="Path to the merged.tsv file",
        type=str,
    )
    parser.add_argument(
        "--threshold",
        default=0.03,
        type=float,
    )
    parser.add_argument(
        "-good", "--scores_path_good", 
        required=True,
        help="The path to the folder where the alignment scores for ACES_final_merged_MQM_good.csv are saved.", 
        action='append', nargs='+'
    )
    parser.add_argument(
        "-bad", "--scores_path_incorrect", 
        required=True,
        help="The path to the folder where the alignment scores for ACES_final_merged_MQM_incorrect.csv are saved.", 
        action='append', nargs='+'
    )
    parser.add_argument(
        "-o", "--out_path",
        required=True,
        help="path to the output tsv file",
        type=str,
    )
    args = parser.parse_args()
    
    assert type(args.threshold) == float and args.threshold >= 0. and args.threshold <=1.

    if "ckpt" in args.model:
        model = load_from_checkpoint(args.model)
    else:
        model_path = download_model(args.model)
        model = load_from_checkpoint(model_path)
    model.encoder.add_span_tokens("<v>", "</v>")

    # -------------------------------------- GOOD TRANSLATIONS --------------------------------------------
    # load the scores
    with open(os.path.join(args.scores_path_good, "ref_scores.json"), "r") as f:
        ref_scores = json.load(f)
    with open(os.path.join(args.scores_path_good, "src_scores.json"), "r") as f:
        src_scores = json.load(f)
    with open(os.path.join(args.scores_path_good, "mt_scores.json"), "r") as f:
        mt_scores = json.load(f)
    with open(os.path.join(args.scores_path_good, "src_ref_scores.json"), "r") as f:
        src_ref_scores = json.load(f)
    gold_span_masks = src_scores["in_span_mask"]
    input_ids = src_scores["input_ids"]
    start_token = model.encoder.tokenizer.encode("<v>")[1]
    end_token = model.encoder.tokenizer.encode("</v>")[1]
    
    gold_spans_input_ids = []
    src_spans_input_ids = []
    ref_spans_input_ids = []
    src_ref_spans_input_ids = []

    # for all the samples, calculate the input ids of the sentences with the generated spans. also for the gold annotated sentneces
    for i in range(len(gold_span_masks)):
        # common in all
        padding_filter = np.array(gold_span_masks[i])!=-1
        input_ids_tmp = np.array(input_ids[i])[padding_filter]
        gold_spans_input_ids.append(add_span_tokens(np.array(gold_span_masks[i])[padding_filter], input_ids_tmp, start_token, end_token))
        # gold_annotated = model.encoder.tokenizer.decode(add_span_tokens(np.array(gold_span_masks[i])[padding_filter], input_ids_tmp, start_token, end_token), skip_special_tokens=True)

        # only src
        src_scores_tmp = np.array(src_scores["scores"][i])[padding_filter]
        src_span_masks = src_scores_tmp > args.threshold
        src_spans_input_ids.append(add_span_tokens(src_span_masks, input_ids_tmp, start_token, end_token))
        # src_annotated = model.encoder.tokenizer.decode(add_span_tokens(src_span_masks, input_ids_tmp, start_token, end_token), skip_special_tokens=True)

        # only ref
        ref_scores_tmp = np.array(ref_scores["scores"][i])[padding_filter]
        ref_span_masks = ref_scores_tmp > args.threshold
        ref_spans_input_ids.append(add_span_tokens(ref_span_masks, input_ids_tmp, start_token, end_token))
        # ref_annotated = model.encoder.tokenizer.decode(add_span_tokens(ref_span_masks, input_ids_tmp, start_token, end_token), skip_special_tokens=True)

        # only src-ref
        src_ref_scores_tmp = np.array(src_ref_scores["scores"][i])[padding_filter]
        src_ref_span_masks = src_ref_scores_tmp > args.threshold
        src_ref_spans_input_ids.append(add_span_tokens(src_ref_span_masks, input_ids_tmp, start_token, end_token))
        # src_ref_annotated = model.encoder.tokenizer.decode(add_span_tokens(src_ref_span_masks, input_ids_tmp, start_token, end_token), skip_special_tokens=True)

        """
        outs = model.encoder.tokenizer.batch_decode([
                                            input_ids[0],
                                            np.array(input_ids[i])[np.array(gold_span_masks[i])==1],
                                            np.array(input_ids[i])[np.array(src_scores["scores"][i])>threshold],
                                            np.array(input_ids[i])[np.array(ref_scores["scores"][i])>threshold],
                                            np.array(input_ids[i])[np.array(src_ref_scores["scores"][i])>threshold],
                                            ],
                                            skip_special_tokens=True)

        print("MT:\t\t", outs[0], "\nGOLD SPAN:\t", outs[1], "\nSRC PRED:\t", outs[2], "\nREF PRED:\t", outs[3], "\nSRC-REF PRED:\t", outs[4])

        print("--------------------------- FULL SPANS ----------------------------------------------------------------")
        print("MT:\t\t", outs[0], "\nGOLD SPAN:\t", gold_annotated, "\nSRC PRED:\t", src_annotated, "\nREF PRED:\t", ref_annotated, "\nSRC-REF PRED:\t", src_ref_annotated)
        """
    # decode to get the spans in string form
    all_spans_decoded = model.encoder.tokenizer.batch_decode(gold_spans_input_ids+src_spans_input_ids+ref_spans_input_ids+src_ref_spans_input_ids, skip_special_tokens=True)

    gold_spans = all_spans_decoded[:len(gold_spans_input_ids)]
    src_spans = all_spans_decoded[len(gold_spans_input_ids):2*len(gold_spans_input_ids)]
    ref_spans = all_spans_decoded[2*len(gold_spans_input_ids):3*len(gold_spans_input_ids)]
    src_ref_spans = all_spans_decoded[3*len(gold_spans_input_ids):4*len(gold_spans_input_ids)]
    
    good_spans = {
        "gold_spans": gold_spans,
        "src_spans": src_spans,
        "ref_spans":ref_spans,
        "src_ref_spans":src_ref_spans
    }

    # -------------------------------------- INCORRECT TRANSLATIONS --------------------------------------------
    # load the scores
    with open(os.path.join(args.scores_path_incorrect, "ref_scores.json"), "r") as f:
        ref_scores = json.load(f)
    with open(os.path.join(args.scores_path_incorrect, "src_scores.json"), "r") as f:
        src_scores = json.load(f)
    with open(os.path.join(args.scores_path_incorrect, "mt_scores.json"), "r") as f:
        mt_scores = json.load(f)
    with open(os.path.join(args.scores_path_incorrect, "src_ref_scores.json"), "r") as f:
        src_ref_scores = json.load(f)
    gold_span_masks = src_scores["in_span_mask"]
    input_ids = src_scores["input_ids"]
    start_token = model.encoder.tokenizer.encode("<v>")[1]
    end_token = model.encoder.tokenizer.encode("</v>")[1]
    
    gold_spans_input_ids = []
    src_spans_input_ids = []
    ref_spans_input_ids = []
    src_ref_spans_input_ids = []

    # for all the samples, calculate the input ids of the sentences with the generated spans. also for the gold annotated sentneces
    for i in range(len(gold_span_masks)):
        # common in all
        padding_filter = np.array(gold_span_masks[i])!=-1
        input_ids_tmp = np.array(input_ids[i])[padding_filter]
        gold_spans_input_ids.append(add_span_tokens(np.array(gold_span_masks[i])[padding_filter], input_ids_tmp, start_token, end_token))
        # gold_annotated = model.encoder.tokenizer.decode(add_span_tokens(np.array(gold_span_masks[i])[padding_filter], input_ids_tmp, start_token, end_token), skip_special_tokens=True)

        # only src
        src_scores_tmp = np.array(src_scores["scores"][i])[padding_filter]
        src_span_masks = src_scores_tmp > args.threshold
        src_spans_input_ids.append(add_span_tokens(src_span_masks, input_ids_tmp, start_token, end_token))
        # src_annotated = model.encoder.tokenizer.decode(add_span_tokens(src_span_masks, input_ids_tmp, start_token, end_token), skip_special_tokens=True)

        # only ref
        ref_scores_tmp = np.array(ref_scores["scores"][i])[padding_filter]
        ref_span_masks = ref_scores_tmp > args.threshold
        ref_spans_input_ids.append(add_span_tokens(ref_span_masks, input_ids_tmp, start_token, end_token))
        # ref_annotated = model.encoder.tokenizer.decode(add_span_tokens(ref_span_masks, input_ids_tmp, start_token, end_token), skip_special_tokens=True)

        # only src-ref
        src_ref_scores_tmp = np.array(src_ref_scores["scores"][i])[padding_filter]
        src_ref_span_masks = src_ref_scores_tmp > args.threshold
        src_ref_spans_input_ids.append(add_span_tokens(src_ref_span_masks, input_ids_tmp, start_token, end_token))
        # src_ref_annotated = model.encoder.tokenizer.decode(add_span_tokens(src_ref_span_masks, input_ids_tmp, start_token, end_token), skip_special_tokens=True)

        """
        outs = model.encoder.tokenizer.batch_decode([
                                            input_ids[0],
                                            np.array(input_ids[i])[np.array(gold_span_masks[i])==1],
                                            np.array(input_ids[i])[np.array(src_scores["scores"][i])>threshold],
                                            np.array(input_ids[i])[np.array(ref_scores["scores"][i])>threshold],
                                            np.array(input_ids[i])[np.array(src_ref_scores["scores"][i])>threshold],
                                            ],
                                            skip_special_tokens=True)

        print("MT:\t\t", outs[0], "\nGOLD SPAN:\t", outs[1], "\nSRC PRED:\t", outs[2], "\nREF PRED:\t", outs[3], "\nSRC-REF PRED:\t", outs[4])

        print("--------------------------- FULL SPANS ----------------------------------------------------------------")
        print("MT:\t\t", outs[0], "\nGOLD SPAN:\t", gold_annotated, "\nSRC PRED:\t", src_annotated, "\nREF PRED:\t", ref_annotated, "\nSRC-REF PRED:\t", src_ref_annotated)
        """
    # decode to get the spans in string form
    all_spans_decoded = model.encoder.tokenizer.batch_decode(gold_spans_input_ids+src_spans_input_ids+ref_spans_input_ids+src_ref_spans_input_ids, skip_special_tokens=True)

    gold_spans = all_spans_decoded[:len(gold_spans_input_ids)]
    src_spans = all_spans_decoded[len(gold_spans_input_ids):2*len(gold_spans_input_ids)]
    ref_spans = all_spans_decoded[2*len(gold_spans_input_ids):3*len(gold_spans_input_ids)]
    src_ref_spans = all_spans_decoded[3*len(gold_spans_input_ids):4*len(gold_spans_input_ids)]
    
    incorrect_spans = {
        "gold_spans": gold_spans,
        "src_spans": src_spans,
        "ref_spans":ref_spans,
        "src_ref_spans":src_ref_spans
    }

    # ---------------------------------------------- SAVING THE SPANS -----------------------------------------------------
    # Save Everything in TSV (similar to merged.tsv).

    # read merged.tsv
    content = read_file(args.dataset)
    # the columns in merged.tsv: ['ID', 'source', 'good-translation', 'incorrect-translation', 'reference',  'phenomena', 'langpair', 'incorrect-translation-annotated', 'annotation-method']
    content = content.reset_index()  # make sure indexes pair with number of rows
    
    # assume the predicted spans are still ordered acc to id?
    samples = {}
    for index, row in content.iterrows():
        idx = int(row["ID"])
        samples[idx] = {
            'source':row['source'],
            'good-translation':row['good-translation'],
            'incorrect-translation':row['incorrect-translation'],
            'reference':row['reference'],
            'phenomena':row['phenomena'],
            'langpair':row['langpair'],
            'incorrect-translation-annotated-goldtruth':row['incorrect-translation-annotated'],
            "annotation-method":row['annotation-method'],
            'good-translation-predicted-span-src':good_spans["src_spans"],
            'good-translation-predicted-span-ref':good_spans["ref_spans"],
            'good-translation-predicted-span-src-ref':good_spans["src_ref_spans"],
            'incorrect-translation-predicted-span-src':incorrect_spans["src_spans"],
            'incorrect-translation-predicted-span-ref':incorrect_spans["ref_spans"],
            'incorrect-translation-predicted-span-src-ref':incorrect_spans["src_ref_spans"],
        }  

    df = pd.DataFrame.from_dict(samples, orient='index', columns=['source', 'good-translation', 'incorrect-translation',
        'reference',  'phenomena', 'langpair', 'incorrect-translation-annotated-goldtruth', 'annotation-method',
        'good-translation-predicted-span-src', 'good-translation-predicted-span-ref', 'good-translation-predicted-span-src-ref',
        'incorrect-translation-predicted-span-src', 'incorrect-translation-predicted-span-ref', 'incorrect-translation-predicted-span-src-ref'])
    df.index.name = 'ID'
    logger.info('Saving to {}'.format(args.out_path+str(args.threshold)+".tsv"))
    df.to_csv(args.out_path+str(args.threshold)+".tsv", sep='\t', index=True, quoting=csv.QUOTE_NONE)