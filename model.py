import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from utils.highway_layer import Highway


class NERModel(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, morph_embedding_dim, word_hidden_size, char_hidden_size, morph_hidden_size, vocab_size, char_vocab_size, morph_vocab_size, tagset_size, word_num_layers, char_num_layers, morph_num_layers, weights_matrix, dropout_prob):

        super(NERModel, self).__init__()
        
        self.weights_matrix = weights_matrix
        self.dropout_prob = dropout_prob
        self.word_hidden_size = word_hidden_size
        self.char_hidden_size = char_hidden_size
        self.morph_hidden_size = morph_hidden_size
        self.word_num_layers = word_num_layers
        self.char_num_layers = char_num_layers
        self.morph_num_layers = morph_num_layers
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.morph_embedding_dim = morph_embedding_dim
        self.vocab_size = vocab_size
        self.char_vocab_size = char_vocab_size 
        self.morph_vocab_size = morph_vocab_size 
        self.tagset_size = tagset_size
        
        self.char_embeddings = nn.Embedding(self.char_vocab_size+1, self.char_embedding_dim)

        self.morph_embeddings = nn.Embedding(self.morph_vocab_size+1, self.morph_embedding_dim)

        self.word_embeddings = nn.Embedding(self.vocab_size+1, self.word_embedding_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(self.weights_matrix))
        self.word_embeddings.weight.requires_grad=False
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_highway = nn.Dropout(p=0.7)

        self.lstm_char = nn.LSTM(self.char_embedding_dim, 
                                 self.char_hidden_size,
                                 num_layers=char_num_layers,
                                 dropout=0.2,
                                 bidirectional=True)

        self.lstm_morph = nn.LSTM(self.morph_embedding_dim, 
                                 self.morph_hidden_size,
                                 num_layers=morph_num_layers,
                                 dropout=0.2,
                                 bidirectional=True)

        self.lstm_word = nn.LSTM(self.word_embedding_dim, 
                                 self.word_hidden_size,
                                 num_layers=word_num_layers,
                                 dropout=0.2,
                                 bidirectional=True)


        self.highway = Highway(self.word_hidden_size*2 + self.char_hidden_size*2 + self.morph_hidden_size*2, 4, f=torch.nn.functional.relu)
    
        self.hidden2tag = nn.Linear(self.word_hidden_size*2 + self.char_hidden_size*2 + self.morph_hidden_size*2, self.tagset_size) 

        self.crf = CRF(self.tagset_size)

        
    def forward(self, pad_seqs, seq_lengths, pad_char_seqs, char_seq_lengths, pad_morph_seqs, morph_seq_lengths, word_hidden, char_hidden, morph_hidden, batch_size):
        # CHAR
        char_embeds = self.char_embeddings(pad_char_seqs)
        char_embeds = char_embeds.mean(-2)
        char_embeds = self.dropout(char_embeds)
        packed_char = pack_padded_sequence(char_embeds, char_seq_lengths)

        char_lstm_out, char_lstm_hidden = self.lstm_char(packed_char, char_hidden)
        char_lstm_out = pad_packed_sequence(char_lstm_out)[0]
    
        # MORPH
        morph_embeds = self.morph_embeddings(pad_morph_seqs)
        morph_embeds = morph_embeds.mean(-2)
        morph_embeds = self.dropout(morph_embeds)
        packed_morph = pack_padded_sequence(morph_embeds, morph_seq_lengths)

        morph_lstm_out, morph_lstm_hidden = self.lstm_morph(packed_morph, morph_hidden)
        morph_lstm_out = pad_packed_sequence(morph_lstm_out)[0]

        # WORD
        word_embeds = self.word_embeddings(pad_seqs).squeeze(dim=2)
        word_embeds = self.dropout(word_embeds)
        packed = pack_padded_sequence(word_embeds, seq_lengths)

        word_lstm_out, word_lstm_hidden = self.lstm_word(packed, word_hidden)
        word_lstm_out = pad_packed_sequence(word_lstm_out)[0]

        output = torch.cat((word_lstm_out, char_lstm_out, morph_lstm_out), dim=2)

        # it takes as input [batch_size, size]
        output = output.permute(1, 0, 2)
        highway_output = self.highway(output)
        highway_output = highway_output.permute(1, 0, 2)
        highway_output = self.dropout_highway(highway_output)

        emissions = self.hidden2tag(highway_output)

        return emissions
    
    
    def init_hidden(self, num_layers, hidden_size, batch_size, device):
        return (autograd.Variable(torch.zeros(2*num_layers, batch_size, hidden_size, device=device)),
                autograd.Variable(torch.zeros(2*num_layers, batch_size, hidden_size, device=device)))