import torch

import io
import gensim
from gensim.models import Word2Vec
from gensim.models.wrappers import FastText
import numpy as np
import pickle

import prepare_data as prepare_data

whole_data_path = '../data/digitoday/digitoday.2014.txt'
whole_data = prepare_data.load_data(whole_data_path)
whole_data = prepare_data.add_start_end_sentence_tokens(whole_data)

custom_embeddings_path = '../data/embeddings/embeddings_150.bin'
embeddings_path = '../data/embeddings/fin-word2vec.bin'
embeddings_path_ft = '../data/embeddings/cc.fi.300.bin'


def create_matrix_custom(embeddings_path, whole_data, dim=300):
    embeddings = Word2Vec.load(embeddings_path)

    word2idx, idx2word, tag2idx, idx2tag, char2idx, idx2char = prepare_data.encode_data(whole_data)

    matrix_len = len(word2idx) + 1
    weights_matrix = np.zeros((matrix_len, dim))
    
    for key, value in word2idx.items():
        try:
            weights_matrix[value] = embeddings[key]
        except KeyError:
            weights_matrix[value] = np.random.normal(scale=0.6, size=(dim, ))

    np.save('../weights/embedding_weights_matrix_custom.npy', weights_matrix)



def create_matrix(embeddings_path, whole_data, dim=300):
    #  shape [4237715, 300]
    embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)

    word2idx, idx2word, tag2idx, idx2tag, char2idx, idx2char = prepare_data.encode_data(whole_data)

    matrix_len = len(word2idx) + 1
    weights_matrix = np.zeros((matrix_len, dim))
    
    for key, value in word2idx.items():
        try:
            weights_matrix[value] = embeddings[key]
        except KeyError:
            weights_matrix[value] = np.random.normal(scale=0.6, size=(dim, ))

    np.save('../weights/embedding_weights_matrix.npy', weights_matrix)

    

def create_matrix_fastTest(embeddings_path, whole_data, dim=300):
    print('loading the embeddings')
    embeddings = FastText.load_fasttext_format(embeddings_path)
    print('successfully loaded the embeddings')


    word2idx, idx2word, tag2idx, idx2tag, char2idx, idx2char = prepare_data.encode_data(whole_data)

    matrix_len = len(word2idx) + 1
    weights_matrix = np.zeros((matrix_len, dim))

    for key, value in word2idx.items():
        try:
            weights_matrix[value] = embeddings[key]
        except KeyError:
            weights_matrix[value] = np.random.normal(scale=0.6, size=(dim, ))

    
    np.save('../weights/embedding_weights_matrix_ft.npy', weights_matrix)

# create_matrix(embeddings_path, whole_data)
create_matrix_fastTest(embeddings_path_ft, whole_data)