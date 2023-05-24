# Span Annotations

## Setting Up the Environment and Downloading the Dataset for the First Time

In the parent directory of the ACES_private folder, create a Python environment: [follow here](https://huggingface.co/docs/datasets/installation)

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

## Running the Jupyter Notebook for Manual Annotations

Activate the environment

    cd folder
    source .env/bin/activate

Start the Jupyter Notebook

    jupyter notebook --port=8889 --no-browser

### Using the Jupyter Notebook to do Manual Annotations

Open the link in the terminal and navigate to ```folder/ACES_private/challenge_set_annotation/annotations.ipynb```. In the notebook:

Run the first cell. It will ask for which phenomena to annotate. 

1. Enter the full name of any phenomena (the phenomena names are listed in the first cell in that huge dictionary called phenomena): for example ```xnli-addition-contradiction```

2. Enter ```test``` : this will load the samples in subset.json file that Chantal collected

This cell will set up everything, if there is already an annotation checkpoint file for that phenomenon (or subset.json) then it will load it. If you don't prefer to continue from the checkpoint but to start from the beginning then run the second cell to reset the annotations for that phenomenon.

Next, run the third cell to start annotating. It will give some suggested annotations, if the annotaions are too long then they are probably wrong. To accept the annotation, press enter. If it is wrong, enter any string to manually annotate, or enter ```skip``` to skip that example (it will leave that example out, and will ask it again if you save a checkpoint and continue from there).

To manually annotate, first copy the incorrect sentence to the textfield and add the <> tokens and press enter. Then do the same for good translation. For some phenomena, we are comparing reference and incorrect, so instead of good translation copy the reference sentence. It will write there which one to use (for some phenomena we either compare the incorrect with good-translation or with reference, so we should make that decision). After entering both incorrect and good translations, it will show the character spans of the manual annotation, and ask you to press ```enter``` to accept it or enter anything else to try again (just like before).

Instead of manually annotating, if you enter ```exit``` it will save a new checkpoint and stop the program. It will also automatucally save the annotations after all the samples are annotated. You can find the checkpoints/annotations in the txt files in ```folder/ACES_private/challenge_set_annotation/manual_annotations/annotated_checkpoint_{phenomena}```.

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

Or run the process_dataset.py script using the detokenize option:

    python ACES_private/challenge_set_annotation/process_dataset.py -d True

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