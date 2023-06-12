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

It automatically loads the examples in the folder/ACES_private/challenge_set_annotation/compare_annotations/annotations/subset.json file. Then you can choose up to 4 other annotated files, which should have the annotations for the samples in the subset.json file. Then it shows each sample in the sample box, and their annotations in the other 4 boxes.