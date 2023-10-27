""" 
Script to Find a Threshold to predict error spans that gievs the best F1 score

span_baseline/baseline1_comet/ACES_alignment_scores/
python ACES_private/span_predictions/scripts/find_threshold.py -m PATH/TO/COMET.ckpt -p span_baseline/baseline1_comet/ACES_alignment_scores/
"""

from tqdm import tqdm
import os, json, argparse
import numpy as np
from comet import download_model, load_from_checkpoint # conda install?

import logging
logger = logging.getLogger('logger')
logging.basicConfig(level=logging.INFO)

from predict_evaluate_spans_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finds a span threshold."
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model or Path to a checkpoint file.",
        type=str,
    )
    parser.add_argument(
        "--metric",
        default="F1",
        type=str,
    )
    parser.add_argument(
        "-p", "--scores_path", 
        required=True,
        help="The path to the folder where the alignment scores for ende-2021-concat.csv are saved.", 
        action='append', nargs='+'
    )
    args = parser.parse_args()
    
    if "ckpt" in args.model:
        model = load_from_checkpoint(args.model)
    else:
        model_path = download_model(args.model)
        model = load_from_checkpoint(model_path)
    
    if args.metric == "F1":
        metric = avg_span_f1
    else:
        logger.error("We only support F1 for now! Exiting")
        exit()
            
    # load the scores
    with open(os.path.join(args.scores_path, "ref_scores.json"), "r") as f:
        ref_scores = json.load(f)
    with open(os.path.join(args.scores_path, "src_scores.json"), "r") as f:
        src_scores = json.load(f)
    with open(os.path.join(args.scores_path, "mt_scores.json"), "r") as f:
        mt_scores = json.load(f)
    with open(os.path.join(args.scores_path, "src_ref_scores.json"), "r") as f:
        src_ref_scores = json.load(f)
    gold_span_masks = src_scores["in_span_mask"]
    input_ids = src_scores["input_ids"]

    # in a for loop calculate the avg F1 accuracies for different thresholds
    thresholds = np.arange(0.005, 0.3, 0.005)
    src_f1, ref_f1, src_ref_f1 = [], [], []

    start_token = model.encoder.tokenizer.encode("<v>")[1]
    end_token = model.encoder.tokenizer.encode("</v>")[1]

    logger.info("Experiments starting..")
    for threshold in tqdm(thresholds):
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
            src_span_masks = src_scores_tmp > threshold
            src_spans_input_ids.append(add_span_tokens(src_span_masks, input_ids_tmp, start_token, end_token))
            # src_annotated = model.encoder.tokenizer.decode(add_span_tokens(src_span_masks, input_ids_tmp, start_token, end_token), skip_special_tokens=True)

            # only ref
            ref_scores_tmp = np.array(ref_scores["scores"][i])[padding_filter]
            ref_span_masks = ref_scores_tmp > threshold
            ref_spans_input_ids.append(add_span_tokens(ref_span_masks, input_ids_tmp, start_token, end_token))
            # ref_annotated = model.encoder.tokenizer.decode(add_span_tokens(ref_span_masks, input_ids_tmp, start_token, end_token), skip_special_tokens=True)

            # only src-ref
            src_ref_scores_tmp = np.array(src_ref_scores["scores"][i])[padding_filter]
            src_ref_span_masks = src_ref_scores_tmp > threshold
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

        # now for all 3 (src, ref, src_ref), calculate the avg F1 scores with given threshold
        src_f1.append(avg_span_f1(src_spans, gold_spans))
        ref_f1.append(avg_span_f1(ref_spans, gold_spans))
        src_ref_f1.append(avg_span_f1(src_ref_spans, gold_spans))
    
    # Plot the Fq values for the thresholds and announce results
    plot([src_f1, ref_f1, src_ref_f1], thresholds)
    threshold = thresholds[np.argmax(src_f1)]
    print(f"The best threshold: {threshold}, the best F1 score: {np.max(src_f1)}")