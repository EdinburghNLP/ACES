# Formatting Functions between Annotation Formats

## Using the merge_annotations script

Example usage: 

If the folder structure looks like this:
```
current_folder
    ├── ACES_private
    |       ├── aces
    |       ├── challenge_set_creation
    |       ├── challenge_set_annotation
    |       |           ├── compare_annotations
    |       |           ├── manual_annotations
    |       ├── score_normalization
    |       └── span_predictions
    |
    ├── manual_annotations 
    |       ├── Error_span_marking
    |       |           ├── annotated_DE.tsv
    |       |           ├── annotated_EN.tsv
    |       |           ├── ...
    |       └── Annotation_SecondDelivery
    |                   ├── annotated_DE.tsv
    |                   ├── annotated_EN.tsv
    |                   └── ...
    |
    └── dataset (download_dataset.py script will create this directory later)
```
Here, Error_span_marking folder has the first delivery of the manual annotations, and Annotation_SecondDelivery has the second delivery where the errors in the annotations in some of the samples were fixed.

According to the given folder structure, we call the [merge_annotations.py](https://github.com/arnisafazla/ACES_private/blob/master/challenge_set_annotation/manual_annotations/merge_annotations.py) script to merge automatic_annotations and manual_annotaions, and then convert all annotations to WMT span annotation format (<w> span <\w>) and save as a tsv file:

    python ACES_private/challenge_set_annotation/manual_annotations/merge_annotations.py -a ACES_private/challenge_set_annotation/annotated.txt -m1 manual_annotations/Error_span_marking -m2 manual_annotations/Annotation_SecondDelivery -o merged.tsv -d dataset
