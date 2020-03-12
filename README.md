# ner-asr

This repository contains a named entity recognition system for Finnish language.

# Requirements
1. pytorch
2. pytorch-crf
3. gensim
4. morfessor

# Download resources
The pretrained word embeddings can be downloaded from the following link: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.bin.gz .

You need to place the embeddings in the `data/embeddings` directory.

# Usage
There are two models trained: `model_lower` and `model_upper`. The first one is trained on lower case data and without punctuation. The second one is trained on data that contains both lower and upper case letter together with punctuation.

To switch between model, change the flag `lowercase_model` in `config/params.py` file.

Use `evaluate_document.py` in order to annotate a new document. 
This script takes two input arguments:

--input - input file to be evaluated

--output - path where the output will be stored

The format of the input document is described in the script
