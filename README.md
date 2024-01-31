# ACES and Span-ACES: Translation Accuracy Challenge Sets for Evaluating Machine Translation Metrics

Recent machine translation (MT) metrics calibrate their effectiveness by correlating with human judgement but without any insights about their behaviour across different error types. Challenge sets are used to probe specific dimensions of metric behaviour but there are very few such datasets and they either focus on a limited number of phenomena or a limited number of language pairs. We introduce ACES, a contrastive challenge set spanning 146 language pairs, aimed at discovering whether metrics can identify 68 translation accuracy errors. These phenomena range from simple alterations at the word/character level to more complex errors based on discourse and real-world knowledge. We conduct a large-scale study by benchmarking ACES on 50 metrics submitted to the WMT 2022 and 2023 metrics shared tasks. We benchmark metric performance, assess their incremental performance over successive campaigns, and measure their sensitivity to a range of linguistic phenomena. We also investigate claims that Large Language Models (LLMs) are effective as MT evaluators by evaluating on ACES. Our results demonstrate that different metric families struggle with different phenomena and that LLM-based methods fail to demonstrate reliable performance. Our analyses indicate that most metrics ignore the source sentence, tend to prefer surface-level overlap and end up incorporating properties of base models which are not always beneficial. We expand ACES to include error span annotations, denoted as SPAN-ACES and we use this dataset to evaluate span-based error metrics showing these metrics also need considerable improvement. Finally, we provide a set of recommendations for building better MT metrics, including focusing on error labels instead of scores, ensembling, designing strategies to explicitly focus on the source sentence, focusing on semantic content and choosing the right base model for representations.

## Download the Dataset

We provide our collection of challenge sets on HuggingFace: [https://huggingface.co/datasets/nikitam/ACES](https://huggingface.co/datasets/nikitam/ACES)

29-01-2024: We have updated the repository to also include Span-ACES. Please look at the dataset repository for more details about the challenge sets.


## Command Line Interface

### Installation

For editable installation:

    pip install -e .

### Scoring Adversarial Examples

Input: One or more TSV files with 'source', 'good-translation', 'incorrect-translation', 'reference', 'phenomena' fields.

You can automatically score adversarial examples by running:

    aces-score -i your_file.tsv ...

Output: One TSV file per input file named 'your_file.scored.tsv' with 'source', 'good-translation', 'incorrect-translation', 'reference', 'phenomena' fields and a 'metric-good' and 'metric-bad' field for every metric that stores the sentence-level scores.

**Currently supported metrics:**

- BLEU
- ChrF
- COMET (needs GPU access)
- COMET-QE (needs GPU access)

### Evaluating Metrics

Input: TSV file with 'source', 'good-translation', 'incorrect-translation', 'reference', 'phenomena' fields and metric scores in 'metric-good' and 'metric-bad' fields.

You can evaluate metrics by running:

    aces-eval -i your_file.tsv ...

Output: STDOUT TSV overview of Kendall Tau scores per file, phenomenon and metric. Use `-p` if you want the script to print a more readable format.

If you want to reproduce the high-level evaluation in our paper (Table 1), you can use the `--print_overview` flag and if you want to compute the ACES score you can use the `--print_aces_score` flag.

For evaluating Span-F1 on Span-ACES, we provide the following script: ``` span_predictions/eval_span.py ```


## Citation

If you use this code, please cite the following papers:

```
    @inproceedings{amrhein-aces-2022,
    title = "{ACES}: Translation Accuracy Challenge Sets for Evaluating Machine Translation Metrics",
    author = {Amrhein, Chantal and
    Moghe, Nikita and
    Guillou, Liane},
    booktitle = "Seventh Conference on Machine Translation (WMT22)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    eprint = {2210.15615}
    }
    

@misc{moghe2024machine,
      title={Machine Translation Meta Evaluation through Translation Accuracy Challenge Sets}, 
      author={Nikita Moghe and Arnisa Fazla and Chantal Amrhein and Tom Kocmi and Mark Steedman and Alexandra Birch and Rico Sennrich and Liane Guillou},
      year={2024},
      eprint={2401.16313},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


```
