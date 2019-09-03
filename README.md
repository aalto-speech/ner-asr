# ner-asr

This repository contains a named entity recognition system for Finnish language.

# Requirements:
1. pytorch
2. pytorch-crf
3. gensim

The finer-data can be downloaded from here: https://github.com/mpsilfve/finer-data/tree/master/data .

Once downloaded, the data needs to be placed in `data/digitoday` directory

The pretrained word embeddings can be downloaded from the following links: bionlp-www.utu.fi/fin-vector-space-models/fin-word2vec.bin and https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.bin.gz .

The first link is for the word2vec embeddings and the second one for the fastText embeddings.

You need to place the embeddings in the `data/embeddings` directory.

Right now I am only using the embeddings of the words that are contained in dataset and drop the rest of the embeddings, therefore I create an embedding matrix containing only the vectors that we need.

The matrix can be created with the following script: `utils/create_embedding_matrix.py` .

Once the matrix is saved, place it in the `wights` directory.

In the `config` directory you can find the parameters used for training the model. You can adjust those before trainig.

Inorder to train and evaluate the model you need to run the `main.py` script.
The model weights will be saved in the `weights` directory.
