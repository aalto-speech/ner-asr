'''

This script takes two input arguments:
--input - input file to be evaluated
--output - path where the output will be stored


The format of the input file should be as following:
**************************************
Kun
Turun
akatemian
ensimmäinen
fysiikan
ja
kasvitieteen
professori
Georgius
Alanus
siirtyi
teologiseen
tiedekuntaan

Kolmen
ehdokkaan
joukosta
pätevimmäksi
katsottiin
Thauvonius
,
ja
hän
saikin
nimityksen
tähän
virkaan
1649
.
**************************************

The output will be in the following format:

**************************************
Kun	O
Turun	B-ORG
akatemian	I-ORG
ensimmäinen	O
fysiikan	O
ja	O
kasvitieteen	O
professori	O
Georgius	B-PER
Alanus	I-PER
siirtyi	O
teologiseen	O
tiedekuntaan	O

Kolmen	O
ehdokkaan	O
joukosta	O
pätevimmäksi	O
katsottiin	O
Thauvonius	B-PER
,	O
ja	O
hän	O
saikin	O
nimityksen	O
tähän	O
virkaan	O
1649	B-DATE
.
**************************************

'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.autograd as autograd

import numpy as np
from argparse import ArgumentParser

from model import NERModel
from train import train
import utils.evaluate as evaluate
import utils.prepare_data as prepare_data
from utils import radam
from config.params import *


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", dest="input",
                        help="input document to be evaluated", metavar="INPUT", required=True)
    parser.add_argument("-o", "--output", dest="output",
                        help="output path", metavar="OUTPUT", required=True)

    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)

    print(device)

    document_path = args.input
    output_path = args.output


    def load_data(data_path):
        words = []
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line != '\n':
                    words.append(line.rstrip())
                else:
                    words.insert(0, '<start>')
                    words.append('<end>')
                    data.append(words)
                    words = []

        return data
    
    target_data = load_data(document_path)


    def combine_data(indexed_data, indexed_char_train, indexed_morph_train, MAX_SEQ_LENGTH):
        res = []
        for seq in range(len(indexed_data)):
            if len(indexed_data[seq]) <= MAX_SEQ_LENGTH:
                res.append((indexed_data[seq], indexed_char_train[seq], indexed_morph_train[seq]))
        return res


    def prepare_sequence(seq, to_ix):
        res = []
        for w in seq:
            res.append([to_ix[w]])
        return autograd.Variable(torch.LongTensor(res))

    def prepare_char_sequence(word, to_ix):
        res = []
        for char in word:
            res.append(to_ix[char])
        return autograd.Variable(torch.LongTensor(res))


    def prepare_morph_sequence(word, to_morph, to_idx):
        res = []
        morphs = to_morph[word]

        for morph in morphs.split(' '):
            res.append(to_idx[morph])
        return autograd.Variable(torch.LongTensor(res))


    def data_to_idx(data, word2idx):
        res = []
        for seq in range(len(data)):
            res.append(prepare_sequence(data[seq], word2idx))
        return res

    def char_to_idx(data, char2idx):
        res = []
        for seq in range(len(data)):
            temp = []
            for w in data[seq]:
                temp.append(prepare_char_sequence(w, char2idx))
            res.append(temp)
        return res


    def morph_to_idx(data, morph2idx, word2morph):
        res = []
        for seq in range(len(data)):
            temp = []
            for w in data[seq]:
                temp.append(prepare_morph_sequence(w, word2morph, morph2idx))
            res.append(temp)
        return res


    def evaluate_document(file, word_num_layers, char_num_layers, morph_num_layers, word_hidden_size, char_hidden_size, morph_hidden_size, batch_size, data, model, idx2word, idx2tag, device):
        with open (file, 'w', encoding='utf-8') as f:
            for sent in data:
                test_sentence = sent[0].to(device)
                chars = sent[1]
                morphs = sent[2]

                word_hidden = model.init_hidden(word_num_layers, word_hidden_size, batch_size, device)
                char_hidden = model.init_hidden(char_num_layers, char_hidden_size, batch_size, device)
                morph_hidden = model.init_hidden(morph_num_layers, morph_hidden_size, batch_size, device)

                pad_char_seqs = prepare_data.pad_subwords(chars).to(device)
                pad_morph_seqs = prepare_data.pad_subwords(morphs).to(device)

                emissions = model(test_sentence, [len(test_sentence)], pad_char_seqs.to(device), [pad_char_seqs.size(0)], pad_morph_seqs.to(device), [pad_morph_seqs.size(0)], word_hidden, char_hidden, morph_hidden, batch_size)

                for i in range(len(test_sentence)):
                    word = idx2word[test_sentence[i].item()]
                    tag = torch.argmax(emissions[i]).item()
                    tag =  idx2tag[torch.argmax(emissions[i]).item()]
                    
                    if word != '<start>' and word != '<end>':
                        f.write(word + '\t' + tag + '\n')
                f.write('\n')


    embeddings_path_ft = 'data/embeddings/cc.fi.300.bin'


    # LOAD INDICES
    word2idx = prepare_data.load_obj('weights/indices/word2idx')
    idx2word = prepare_data.load_obj('weights/indices/idx2word')
    tag2idx = prepare_data.load_obj('weights/indices/tag2idx')
    idx2tag = prepare_data.load_obj('weights/indices/idx2tag')
    char2idx = prepare_data.load_obj('weights/indices/char2idx')
    morph2idx = prepare_data.load_obj('weights/indices/morph2idx')
    idx2morph = prepare_data.load_obj('weights/indices/idx2morph')
    word2morph = prepare_data.load_obj('weights/indices/word2morph')

    # load embedding matrix
    weights_matrix = np.load('weights/embedding_weights_matrix_ft.npy')

    
    def remove_oov(target_data, idx2word, word2morph):
        data = []
        temp = []
        for sent in target_data:
            for word in sent:
                if word in idx2word.values() and word in word2morph.keys():
                    temp.append(word)
            data.append(temp)
            temp = []
        return data

    target_data = remove_oov(target_data, idx2word, word2morph)

    indexed_data_target = data_to_idx(target_data, word2idx)
    indexed_char_target = char_to_idx(target_data, char2idx)
    indexed_morph_target = morph_to_idx(target_data, morph2idx, word2morph)
    data_target = combine_data(indexed_data_target, indexed_char_target, indexed_morph_target, MAX_SEQ_LENGTH)


    # initialize the model
    model = NERModel(word_embedding_dim, char_embedding_dim, morph_embedding_dim, word_hidden_size, char_hidden_size, morph_hidden_size, len(word2idx), 
                len(char2idx), len(morph2idx), len(tag2idx)+1, word_num_layers, char_num_layers, morph_num_layers, weights_matrix, dropout_prob).to(device)

    # load the model
    model.load_state_dict(torch.load('weights/model.pt'))

    model.eval()

    batch_size = 1

    print('Processing the document')
    evaluate_document(output_path, word_num_layers, char_num_layers, morph_num_layers, word_hidden_size, char_hidden_size, morph_hidden_size, batch_size, data_target, model, idx2word, idx2tag, device)
    print('Done')