#!/usr/bin/env bash
# -- coding: utf-8 --

source .env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ./ACES_private/challenge_set_annotation/requirements.txt
python3 ./ACES_private/challenge_set_annotation/download_dataset.py
deactivate