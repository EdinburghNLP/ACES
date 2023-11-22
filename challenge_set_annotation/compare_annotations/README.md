# Comparing/Visualizing the Annotations on Browser

Folder structure:
```
folder
    ├── .env
    ├── ACES_private
    └── dataset
```

## Starting the python server

    cd folder/ACES_private/challenge_set_annotation/compare_annotations
    sudo python3 -m http.server 80

Then on the browser (works with Google chrome, I am not sure if it would work with other browsers) go to this address:

    http://localhost/test.html

## Using the tool

There are two attach file inputs, to load subset.json (or subset.tsv) and up to 6 annotation files (json or tsv). 

subset.tsv or subset.json should be any file that has all the samples we want to view (for example you can just load there the /ACES_private/challenge_set_annotation/annotated.txt file, which has all the samples which are annotated manually, but it slows it down a little because too large.)

For example to compare the subset_a.tsv that Chantal compiled, use the first file attach button to choose the subset_a.tsv, then use the second file attach button to choose subset_a_annotated_automatic.tsv, subset_a_annotated_nikita.tsv.. up to 6 files!

After navigating and uploading the files, use Next and Back buttons to see the annotations on the incorrect translations for each sample.