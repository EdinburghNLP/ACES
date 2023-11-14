# Predict Error-Spans Using [COMET - Explainable Metrics](https://arxiv.org/pdf/2305.11806.pdf) and Evaluate Them

## 1. Setting Up the [COMET - Explainable Metrics](https://arxiv.org/pdf/2305.11806.pdf)
```bash
pip install --upgrade pip 
pip install git+https://github.com/Unbabel/COMET.git@explainable-metrics
```

Download the models from HF Hub:
- [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da)
- [`Unbabel/wmt22-unite-da`](https://huggingface.co/Unbabel/wmt22-unite-da)

## 2. Convert ACES final format to [MQM format](https://github.com/Unbabel/COMET/tree/explainable-metrics) used in the explainable COMET paper

ACES format:
```
        {'source': "Proper nutritional practices alone cannot generate elite performances, but they can significantly affect athletes' overall wellness.",
        'good-translation': 'Las prácticas nutricionales adecuadas por sí solas no pueden generar rendimiento de élite, pero pueden afectar significativamente el bienestar general de los atletas.',
        'incorrect-translation': 'Las prácticas nutricionales adecuadas por sí solas no pueden generar rendimiento de élite, pero pueden afectar significativamente el bienestar general de los jóvenes atletas.',
        'reference': 'No es posible que las prácticas nutricionales adecuadas, por sí solas, generen un rendimiento de elite, pero puede influir en gran medida el bienestar general de los atletas .',
        'phenomena': 'addition',
        'langpair': 'en-es',
        'incorrect-translation-annotated': 'Las prácticas nutricionales adecuadas por sí solas no pueden generar rendimiento de élite, pero pueden afectar significativamente el bienestar general de los <v>jóvenes</v> atletas.',
        'annotation-method': 'annotate_word'}
```

MQM format:
```
        Columns:    src,    mt,                     ref,        score,  system,             lp,         segid,  annotation  incorrect
        Row:    source  incorrect-translation   reference   None    annotation-method   langpair    ID      incorrect-translation-annotated True/False
```
We add 1 new field in the MQM format: 
* incorrect: True if that is incorrect translation and False if correct.

Assuming the file structure:
```
current_folder
    ├── ACES_private
    |       ├── aces
    |       ├── challenge_set_creation
    |       ├── challenge_set_annotation
    |       ├── score_normalization
    |       └── span_predictions
    |                   ├── format_utilities.py
    |                   ├── to_MQM_format.py
    |                   └── scripts/
    |
    ├── span_baseline
    |       ├── baseline_1_comet
    |       |           ├── ende-2021-concat.csv
    |       |           ├── ACES_final_merged_MQM_good.csv !!!!!
    |       |           ├── ACES_final_merged_MQM_incorrect.csv !!!!!
    |       |           └── ...
    |       └── baseline_2_trained
    |
    └── merged.tsv
```

Run:

    python ACES_private/span_predictions/to_MQM_format.py -d merged.tsv -o span_baseline/baseline1_comet

## 3. Saving COMET Alignmnet scores

The code ([ACES_private/span_predictions/scripts/save_comet_alignments.py](https://github.com/arnisafazla/ACES_private/blob/master/span_predictions/scripts/save_COMET_alignments.py), adapted from [explain_comet.py](https://github.com/Unbabel/COMET/blob/explainable-metrics/explainable-metrics/explain_comet.py)) finds an alignment between these sentences and saves them as np arrays: \\
    * source and translation \\
    * reference and translation \\
    * source+reference and translation
    
using the attention head values extracted from COMET. This scores every subword in the translated sentence, where a higher score means the possibility that the subword is in the error span is higher.

Make sure to run it with GPUs.

We extract the alignment scores for 2 datasets: 

### a. The data used in [COMET - Explainable Metrics](https://arxiv.org/pdf/2305.11806.pdf):
We use this data as our development set to find a good threshold for span predictions. It can be downloaded from the following links:
- [ZhEn MQM](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/acl2023/zhen-2021-concat.csv): WMT 2021 MQM annotations.
- [ZhEn SMAUG](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/acl2023/zhen-smaug.csv): SMAUG Perturbations.
- [EnDe MQM](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/acl2023/ende-2021-concat.csv): WMT 2021 MQM annotations.
- [EnRu MQM](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/acl2023/enru-2021-concat.csv): WMT 2021 MQM annotations.

### b. ACES in MQM format (See Part 2)
We need the alignment scores to predict error spans on ACES.

Run this to we save the alignment scores: 

        python ACES_private/span_predictions/scripts/save_comet_alignments.py -m {model_path} -t {WMT_data_path} --batch_size 8 -o {output_path}

Where 
* ```model_path``` is the path to the model.ckpt in the folder downloaded from [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da)
* ```WMT_data_path``` is either the path to development set or to the ACES in MQM format (for example ```span_baseline/baseline1_comet/ACES_final_merged_MQM.csv```)
* output_path is a folder, for example ```span_baseline/baseline1_comet/ACES_alignment_scores/```

## 4. Find a Threshold Using the Development set [EnDe MQM](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/acl2023/ende-2021-concat.csv): WMT 2021 MQM annotations.

        python ACES_private/span_predictions/scripts/find_threshold.py -m PATH/TO/COMET.ckpt -p span_baseline/baseline1_comet/ACES_alignment_scores_good/

It will print the threshold with the best F1 score

## 5. Save spans for ACES
First, we need to save alignment score for all the samples, incorrect and good translations separately (using GPUs).

        python ACES_private/span_predictions/scripts/save_comet_alignments.py -m {model_path} -t span_baseline/baseline1_comet/ACES_final_merged_MQM_good.csv --batch_size 8 -o span_baseline/baseline1_comet/ACES_alignment_scores_good/
        python ACES_private/span_predictions/scripts/save_comet_alignments.py -m {model_path} -t span_baseline/baseline1_comet/ACES_final_merged_MQM_incorrect.csv --batch_size 8 -o span_baseline/baseline1_comet/ACES_alignment_scores_incorrect/

Then the folder structure of ```span_baseline/baseline1_comet``` looks like this with the outputs:

```
baseline_1_comet
    ├── ende-2021-concat.csv
    ├── ACES_final_merged_MQM_good.csv
    ├── ACES_final_merged_MQM_incorrect.csv
    ├── ACES_alignment_scores_incorrect/
    └── ACES_alignment_scores_good
                ├── mt_scores.json
                ├── ref_scores.json
                ├── src_ref_scores.json
                └── src_scores.json
```

Then, run the save_span_predictions.py to generate and save the predicted spans.

        python ACES_private/span_predictions/save_span_predictions.py -m {MODEL_PATH} -d merged.tsv --threshold 0.03 --good span_baseline/baseline1_comet/ACES_alignment_scores_good/ --bad span_baseline/baseline1_comet/ACES_alignment_scores_incorrect/ -o span_baseline/baseline1_comet/ACES_predicted_spans_

## 6. The Example Pipeline for 3, 4, 5 (for Saving Error Spans on ACES using COMET/UNITE Embeddings)

After generating the files ACES_final_merged_MQM_good.csv and ACES_final_merged_MQM_incorrect.csv (see 2), you can see the example of how to find the thresholds and to generate and save the spans in the notebook 
[current_folder/ACES_private/span_predictions/ACES_COMET_UNITE_span_baseline.ipynb](https://github.com/arnisafazla/ACES_private/blob/master/span_predictions/ACES_COMET_UNITE_span_baseline.ipynb). 