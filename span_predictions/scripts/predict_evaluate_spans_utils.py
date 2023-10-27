"""
Functions to calculate F1 score given 
"""
import re
import numpy as np
import matplotlib.pyplot as plt


""" inputs:
    binary_mask: binary array, True for the subwords that are included in the span and False for others
    integer_array: the input_ids, which is the input sentence encoded by a tokenizer and split into subwords
    start_token: tokenizer-encode("<v>")
    end_token: tokenizer-encode("</v>")
"""
def add_span_tokens(binary_mask, integer_array, start_token, end_token):
    modified_integer_array = []

    # Handle the case when binary_mask starts with 1
    if binary_mask[0] == 1:
        modified_integer_array.append(start_token)
        modified_integer_array.append(integer_array[0])
    else:
        modified_integer_array.append(integer_array[0])

    for i in range(1, len(binary_mask)):
        if binary_mask[i] != binary_mask[i - 1]:
            if binary_mask[i] == 1:  # Change from 0 to 1
                modified_integer_array.append(start_token)
            else:  # Change from 1 to 0
                modified_integer_array.append(end_token)
            modified_integer_array.append(integer_array[i])
        else:
            modified_integer_array.append(integer_array[i])

    # Handle the case when binary_mask ends with 1
    if binary_mask[-1] == 1:
        modified_integer_array.append(end_token)
    else:
        modified_integer_array.append(integer_array[-1])

    return modified_integer_array

# Calculate F1 score for prediction string and ground truth string
def span_f1(prediction, ground_truth):
    pattern = r'<v>(.*?)</v>'

    predicted_spans = re.findall(pattern, prediction)
    ground_truth_spans = re.findall(pattern, ground_truth)

    tp = len(set(predicted_spans) & set(ground_truth_spans))
    fp = len(set(predicted_spans) - set(ground_truth_spans))
    fn = len(set(ground_truth_spans) - set(predicted_spans))
    # calculate precision, recall and f1
    if tp == 0 and fp == 0 and fn == 0:
        # Both ground truth and prediction sets are empty
        f1 = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1

# Calculate average F1 score for given 2 lists of strings
def avg_span_f1(predictions, ground_truths):
    f1_scores = [span_f1(predictions[i], ground_truths[i]) for i in range(len(predictions))]
    return np.average(f1_scores)

def plot(f1_scores, thresholds):
    src = f1_scores[0]
    ref = f1_scores[1]
    src_ref = f1_scores[2]
    plt.plot(thresholds, src, '-o', label='src')
    plt.plot(thresholds, ref, '-s', label='ref')
    plt.plot(thresholds, src_ref, '-^', label='src_ref')
    plt.xlabel('Thresholds')
    plt.ylabel('F1 Scores')
    plt.title('F1 scores per Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()