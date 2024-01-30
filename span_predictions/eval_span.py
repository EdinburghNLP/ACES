import pandas as pd
import re
import csv
import sys

def span_f1(predicted_spans, ground_truth_spans):
    """
    Calculate span f1 score per example
    """
    tp = len(set(predicted_spans) & set(ground_truth_spans))
    fp = len(set(predicted_spans) - set(ground_truth_spans))
    fn = len(set(ground_truth_spans) - set(predicted_spans))
    
    if len(ground_truth_spans) == 0 and len(predicted_spans) == 0:
        return 1.
    
    #calculate precision, recall and f1
    if tp == 0.:
        precision, recall = 0., 0,
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    if precision == 0. or recall == 0.:
        return 0.
    else:
        return 2 * precision * recall / (precision + recall)


def get_spans_from_sent(sentence):
    """
    Extracting spans from a sentence. The spans have to be in the format <v>span</v>
    Currently, it includes punctuations within the span
    """
    sentence = str(sentence)
    spans = []
    pattern = r'<v>(.*?)</v>'
    spans_all = re.findall(pattern, sentence)
    for span in spans_all:
        span = span.strip()
        if len(span) > 0:
            spans.append(span)
    return spans

def read_file(filename: str) -> pd.DataFrame:
    '''
    Read TSV file and return as pandas DataFrame
    '''
    return pd.read_csv(filename, sep='\t', quoting=csv.QUOTE_NONE)

def len_heuristic(good_spans, bad_spans):
    '''
    Heuristic to assign scores to good and bad spans based on their length. This is not applicable to calculating span-extraction F1 but may be useful for the contrastive evaluation of ACES with spans
    '''
    score_g = []
    score_b = []
    for i in range(len(good_spans)):
        if len(good_spans[i]) < len(bad_spans[i]):
            score_g.append(60)
            score_b.append(50)
        else:
            score_g.append(10)
            score_b.append(50)
    return score_g, score_b



data = read_file(sys.argv[1])
'''
This assumes that the pandas file contains a prediction column with <v>span</v> format for both good translation and incorrect translation
'''

span_f1s = []
good_scores = []
bad_scores = []
for prediction_good, prediction_incorrect, ground_truth in zip(data['good-translation-prediction'], data['incorrect-translation-prediction'], data['incorrect-translation-annotated']):
    prediction_incorrect_spans = get_spans_from_sent(prediction_incorrect)
    prediction_good_spans = get_spans_from_sent(prediction_good)
    ground_truth_spans = get_spans_from_sent(ground_truth)
    score_g, score_b = len_heuristic(prediction_good_spans, prediction_incorrect_spans)
    span_f1s.append(span_f1(prediction_incorrect_spans, ground_truth))
    good_scores.extend(score_g)
    bad_scores.extend(score_b)

print("Span F1: ", sum(span_f1s)/len(span_f1s))

assert len(good_scores) == len(bad_scores)
data['metric-good'] = good_scores
data['metric-bad'] = bad_scores

#Save the dataframe and then evaluate using aces-eval as instructed in the README
