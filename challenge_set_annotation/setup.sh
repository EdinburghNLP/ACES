#!/usr/bin/env bash
# -- coding: utf-8 --

source ../../.env/bin/activate
python -m pip install --upgrade pip
pip install -r ../ACES_private/challenge_set_annotation/requirements.txt
python3 ../ACES_private/challenge_set_annotation/download_dataset.py