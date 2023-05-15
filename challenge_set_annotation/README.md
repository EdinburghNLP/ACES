# Span Annotations

## Setting Up the Environment and Downloading the Dataset for the First Time

Create Python environment and install datasets library: [follow here](https://huggingface.co/docs/datasets/installation)

Folder structure:
```
folder
    ├── .env (follow https://huggingface.co/docs/datasets/installation) 
    ├── ACES_private
    └── dataset (download_dataset.py script will create this directory later)
```


Go to folder

    cd folder

Setup the environment and download the dataset

    bash ./ACES_private/challenge_set_annotation/setup.sh

## Running the Jupyter Notebook

Activate the environment

    cd folder
    source .env/bin/activate

Start the Jupyter Notebook

    jupyter notebook --port=8889 --no-browser

## Running the whole annotation script

Run test cases 

    cd folder
    source .env/bin/activate
    cd folder/ACES_private/challenge_set_annotation/
    python -m unittest discover

Run the process_dataset.py script and generate annotated.txt and stats.txt files

    cd folder
    source .env/bin/activate
    python ACES_private/challenge_set_annotation/process_dataset.py

In folder/annotated.txt, the format looks like this:

```
{"id": {'source': "source sentence",
    'good-translation': 'good translation',
    'incorrect-translation': 'incorrect translation',
    'reference': 'reference sentence',
    'phenomena': 'ordering-mismatch',
    'langpair': 'en-de',
    'annotation': 
        [{'in_good': {'token_index': [...],
        'character_span': (30, 80),
        'token': 'the changed/matched token in the good-translation'},
    'in_bad': {'token_index': [...],
        'character_span': (29, 80),
        'token': 'the changed/matched token in the good-translation'}},
    {'in_good': {'token_index': [...],
        'character_span': (44, 80),
        'token': '...'},
    'in_bad': {'token_index': [...],
        'character_span': (29, 65),
        'token': '...'}}],
    'method': 'the method used to generate the annotation.'}
}
```

And in stats.txt, the format looks like this:

```
{'error': [(id, 'zh-en'), ...],      # samples that caused errors
 'no_change': [(id, 'zh-en'), ...],  # samples whose annotations are empty
 'other': [(id, 'zh-en'), ...],      # samples that caused other problems
 'success': 969,                     # no. of samples (probably) correctly classified
 'too_long': [(id, 'zh-en'), ...],   # samples whose annotations are too long (over 50 characters)
 'total': 1000}                      # total no. of samples in this category (phenomenon)
```