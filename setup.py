#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

setup(
    name='aces',
    version='1.0.0',
    author='Nikita Moghe, Liane Guillou, Chantal Amrhein',
    author_email='nikita.moghe@ed.ac.uk,liane.guillou@ed.ac.uk,amrhein@cl.uzh.ch',
    description='a tool to create and score adversarial translation hypothesis pairs and evaluate metrics',
    long_description=readme,
    packages=[
        'aces',
    ],
    install_requires=[
        'jsonargparse',
        'names',
        'pandas',
        'pytest',
        'pint',
        'quantulum3',
        'sacrebleu',
        'spacy',
        'textdistance',
        'unbabel-comet',
    ],
    url='https://github.com/nikitacs16/build-it-break-it-22',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
    ],
    entry_points={
        'console_scripts': [
            'aces-score = aces.cli.score:score',
            'aces-eval = aces.cli.evaluate:evaluate',
        ],
    },
    python_requires='>=3.8',
    license='MIT',
)
