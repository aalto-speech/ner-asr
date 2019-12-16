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

You can download the model weights from the following link: https://drive.google.com/open?id=1eSfSh6Ch8P96bTde5-uSechayfx4oECV

The weights should be placed in a `weights` directory

# Usage
In order to evaluate the system on the Digitoday dataset, run `main_digitoday.py` script.

In order to evaluate the system on the Parliament dataset, run `main_parliament.py` script. If you want to evaluate it only on the entities that were found by Lingsoft, set the parameter `full_asr_evaluation` to `False` in the `config/config.py` file.

In order to evaluate the system on the Pressiklubi dataset, run `main_pressiklubi.py` script. If you want to evaluate it only on the entities that were found by Lingsoft, set the parameter `full_asr_evaluation` to `False` in the `config/config.py` file.
