#!/usr/bin/env bash
# -- coding: utf-8 --

# folder structure is like this:

# folder
# -> .env (follow https://huggingface.co/docs/datasets/installation)
# -> ACES_private (github repo)
# -> dataset (download_dataset.py script will create this directory later)

# cd folder

source .env/bin/activate
python -m pip install --upgrade pip
pip install -r ./ACES_private/challenge_set_annotation/requirements.txt
python3 ./ACES_private/challenge_set_annotation/download_dataset.py
python3 ./ACES_private/challenge_set_annotation/process_dataset.py

deactivate