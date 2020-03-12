import torch
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import string


def load_data(data_path):
    data = []
    word = []
    tag = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                word.append(line.split('\t')[0])
                tag.append(line.split('\t')[1].rstrip())
            else:
                data.append((word, tag))
                word = []
                tag = []

    return data


def load_data_morphs(data_path):
    morphs = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if '\n' in line:
                line = line.replace('\n', '')
            morphs.append(line)

    return morphs


def load_whole_data(train_path, dev_path, test_path, wiki_path):
    data = []
    word = []
    tag = []

    data_sources = [train_path, dev_path, test_path, wiki_path]
    for source in data_sources:
        with open(source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line != '\n':
                    word.append(line.split('\t')[0])
                    tag.append(line.split('\t')[1])
                else:
                    data.append((word, tag))
                    word = []
                    tag = []

    return data


def word_to_morph(data_morphs):
    word2morph = {}
    word2morph['<start>'] = '<start>'
    word2morph['<end>'] = '<end>'

    for seq in data_morphs:
        word = ''
        segments = seq.split(' ')
        segments_with_boundaries = add_subword_boundaries(segments)

        for segment in segments:
            word += segment
        word2morph[word] = segments_with_boundaries

    return word2morph


def prepare_char_sequence(word, to_ix):
    res = []
    for char in word:
        res.append(to_ix[char])

    return autograd.Variable(torch.LongTensor(res))


def prepare_morph_sequence(word, to_morph, to_idx):
    res = []
    morphs = to_morph[word]

    for morph in morphs.split(' '):
    # for morph in morphs:
        res.append(to_idx[morph])

    return autograd.Variable(torch.LongTensor(res))


def prepare_sequence(seq, to_ix, embeddings):
    res = []
    for w in seq:
        # res.append([to_ix[w]])
        try:
            res.append(embeddings[w])
        except:
            res.append(np.random.normal(scale=0.6, size=(300, )))
    res = autograd.Variable(torch.FloatTensor(res))

    return res


def data_to_idx(data, word2idx, embeddings):
    res = []
    for seq in range(len(data)):
        res.append(prepare_sequence(data[seq][0], word2idx, embeddings))
    return res


def prepare_target(seq, to_ix):
    res = []
    for w in seq:
        res.append([to_ix[w]])

    return autograd.Variable(torch.LongTensor(res))



def char_to_idx(data, char2idx):
    res = []
    for seq in range(len(data)):
        temp = []
        for w in data[seq][0]:
            temp.append(prepare_char_sequence(w, char2idx))
        res.append(temp)
        
    return res


def morph_to_idx(data, morph2idx, word2morph):
    res = []
    for seq in range(len(data)):
        temp = []
        for w in data[seq][0]:
            temp.append(prepare_morph_sequence(w, word2morph, morph2idx))
        res.append(temp)

    return res


def tag_to_idx(data, tag2idx):
    res = []
    for seq in range(len(data)):
        res.append(prepare_target(data[seq][1], tag2idx))

    return res


def combine_data(indexed_data, indexed_tag, indexed_char_train, indexed_morph_train, MAX_SEQ_LENGTH):
    res = []
    for seq in range(len(indexed_data)):
        if len(indexed_data[seq]) <= MAX_SEQ_LENGTH:
            res.append((indexed_data[seq], indexed_tag[seq], indexed_char_train[seq], indexed_morph_train[seq]))

    return res


# convert words and tags to indices
def encode_data(whole_data):
    word2idx = {}
    idx2word = {}
    tag2idx = {}
    idx2tag = {}
    char2idx = {}
    idx2char = {}


    for sent, tags in whole_data:
        for tag in tags:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx) + 1
                idx2tag[len(idx2tag) + 1] = tag
        
        for word in sent:
            if word not in word2idx:            
                word2idx[word] = len(word2idx) + 1
                idx2word[len(idx2word) + 1] = word
            for char in word:
                if char not in char2idx:
                    char2idx[char] = len(char2idx) + 1
                    idx2char[len(idx2char) + 1] = char

    return word2idx, idx2word, tag2idx, idx2tag, char2idx, idx2char


def encode_data_morphs(morph_data):
    morph2idx = {}
    idx2morph = {}

    morph2idx['<start>'] = 1
    idx2morph[1] = '<start>'
    morph2idx['<end>'] = 1
    idx2morph[1] = '<end>'

    for morphs in morph_data:
        subwords = morphs.split(' ')
        morphs = add_subword_boundaries(subwords)

        for morph in morphs.split(' '):
            if morph not in morph2idx:
                morph2idx[morph] = len(morph2idx) + 1
                idx2morph[len(idx2morph) + 1] = morph

    return morph2idx, idx2morph


# convert to lower case
def to_lower(data):
    lower_data = []
    for seq in data:
        sentence = seq[0]
        tags = seq[1]
        lower_sent = [s.lower() for s in sentence]
        lower_data.append((lower_sent, tags))
    return lower_data


# remove punctuation
def remove_punct(data):
    new_data = []
    for seq in data:
        sentence = seq[0]
        tags = seq[1]

        clean_sent = []
        clean_tag = []
        for s in range(len(sentence)):
            if sentence[s] not in string.punctuation:
                clean_sent.append(sentence[s])
                clean_tag.append(tags[s])
        new_data.append((clean_sent, clean_tag))
    return new_data


# add <start> and <end> tokens
def add_start_end_sentence_tokens(data):
    for sent in range(len(data)):
        data[sent][0].insert(0, '<start>')
        data[sent][0].append('<end>')
        data[sent][1].insert(0, 'O')
        data[sent][1].append('O')
    return data


# add subword boundaries, example: liiketoiminta+ +yksikkö
def add_subword_boundaries(subwords):
    res = ''
    if len(subwords) == 1:
        res = subwords[0]
    else:
        for i in range(len(subwords)):
            if i == 0:
                res += subwords[i] + '+'
            elif i < len(subwords) - 1:
                res += ' +' + subwords[i] + '+'
            elif i == len(subwords) - 1:
                res += ' +' + subwords[i]

    return res


# extra data to be removed so that it can be divided in equal batches
def remove_extra(data, batch_size):
    extra = len(data) % batch_size
    data = data[:-extra][:]
    return data
    
# pad chars and morphs
# idx = 2 for chars, 3 for morphs
def pad_subwords(subwords):   
    subword_lengths = []
    for seq in subwords:
        subword_lengths.append(seq.size(0))
    max_subword_length = max(subword_lengths)
    
    for seq in range(len(subwords)):
        pad_size = max_subword_length - subwords[seq].size(0)
        pad_tensor = torch.zeros([pad_size], dtype=torch.int64)

        if pad_size != 0:
            subwords[seq] = torch.cat((subwords[seq], pad_tensor), 0)
    

    pad_subword_seqs = torch.stack(subwords)
    pad_subword_seqs = pad_subword_seqs.unsqueeze(1)
    
    return pad_subword_seqs


def collate(list_of_samples):
    # sort a list by sequence length
    list_of_samples.sort(key=lambda x: len(x[0]), reverse=True)

    input_seqs, output_seqs, char_seqs, morph_seqs = zip(*list_of_samples)
    input_seq_lengths = [len(seq) for seq in input_seqs]
    output_seq_lengths = [len(seq) for seq in output_seqs]
    char_seq_lengths = [len(seq) for seq in char_seqs]
    morph_seq_lengths = [len(seq) for seq in morph_seqs]

    padding_value = 0
    pad_input_seqs = pad_sequence(input_seqs, padding_value=padding_value)

    # pad chars within sentence
    pad_chars = []
    for i in range(len(char_seqs)):
        pad_chars.append(pad_sequence(char_seqs[i], padding_value=padding_value))

    # pad morphs within sentence
    pad_morphs = []
    for i in range(len(morph_seqs)):
        pad_morphs.append(pad_sequence(morph_seqs[i], padding_value=padding_value))

    # pad chars within batch
    # pad word level
    sentence_lengths = []
    for seq in pad_chars:
        sentence_lengths.append(seq.size(1))
    max_sentence_length = max(sentence_lengths)

    for seq in range(len(pad_chars)):
        pad_size = max_sentence_length - pad_chars[seq].size(1)
        pad_tensor = torch.zeros([pad_chars[seq].size(0), pad_size], dtype=torch.int64)
        if pad_size != 0:
            pad_chars[seq] = torch.cat((pad_chars[seq], pad_tensor), 1)


    # pad char level
    char_lengths = []
    for seq in pad_chars:
        char_lengths.append(seq.size(0))
    max_char_length = max(char_lengths)

    for seq in range(len(pad_chars)):
        pad_size = max_char_length - pad_chars[seq].size(0)
        pad_tensor = torch.zeros([pad_size, pad_chars[seq].size(1)], dtype=torch.int64)
        if pad_size != 0:
            pad_chars[seq] = torch.cat((pad_chars[seq], pad_tensor), 0)
    

    pad_char_seqs = torch.stack(pad_chars)
    pad_char_seqs = pad_char_seqs.permute(2, 0, 1)



    # pad morphs within batch
    # pad word level
    sentence_lengths = []
    for seq in pad_morphs:
        sentence_lengths.append(seq.size(1))
    max_sentence_length = max(sentence_lengths)

    for seq in range(len(pad_morphs)):
        pad_size = max_sentence_length - pad_morphs[seq].size(1)
        pad_tensor = torch.zeros([pad_morphs[seq].size(0), pad_size], dtype=torch.int64)
        if pad_size != 0:
            pad_morphs[seq] = torch.cat((pad_morphs[seq], pad_tensor), 1)


    # pad morph level
    morph_lengths = []
    for seq in pad_morphs:
        morph_lengths.append(seq.size(0))
    max_morph_length = max(morph_lengths)

    for seq in range(len(pad_morphs)):
        pad_size = max_morph_length - pad_morphs[seq].size(0)
        pad_tensor = torch.zeros([pad_size, pad_morphs[seq].size(1)], dtype=torch.int64)
        if pad_size != 0:
            pad_morphs[seq] = torch.cat((pad_morphs[seq], pad_tensor), 0)
    

    pad_morph_seqs = torch.stack(pad_morphs)
    pad_morph_seqs = pad_morph_seqs.permute(2, 0, 1)


    # pad output sequences
    pad_output_seqs = []
    for i in output_seqs:
        padded = i.new_zeros(max(output_seq_lengths) - i.size(0))
        pad_output_seqs.append(torch.cat((i, padded.view(-1, 1)), dim=0))

    pad_output_seqs = torch.stack(pad_output_seqs)
    pad_output_seqs = pad_output_seqs.permute(1, 0, 2)

    return pad_input_seqs, input_seq_lengths, pad_output_seqs, output_seq_lengths, pad_char_seqs, char_seq_lengths, pad_morph_seqs, morph_seq_lengths