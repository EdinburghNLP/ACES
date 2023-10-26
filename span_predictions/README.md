# Converting the Format

# Setting Up and Running the Explainable Comet and Unite Code

1. Get the [COMET - Explainable Metrics](https://arxiv.org/pdf/2305.11806.pdf)
```bash
pip install --upgrade pip 
pip install git+https://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@github.com/Unbabel/COMET.git@explainable-metrics
```

Download the models from HF Hub:
- [`Unbabel/wmt22-comet-da`](https://huggingface.co/Unbabel/wmt22-comet-da)
- [`Unbabel/wmt22-unite-da`](https://huggingface.co/Unbabel/wmt22-unite-da)

2. Convert ACES final format to [MQM format](https://github.com/Unbabel/COMET/tree/explainable-metrics) used in the explainable COMET paper

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
We add 2 new fields in the MQM format: 
* incorrect: True if that is incorrect translation and False if correct.
* annotation-method: For further analysis

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
    |       ├── baseline1_comet
    |       |           ├── ende-2021-concat.csv
    |       |           ├── ACES_final_merged_MQM.csv !!!!!
    |       |           └── ...
    |       ├── baseline2_trained
    |
    └── merged.tsv
```

Run:

    python to_MQM_format.py -d merged.tsv -o span_baseline/baseline1_comet/ACES_final_merged_MQM.csv