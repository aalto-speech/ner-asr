import sys
sys.path.insert(0,'../..')

import prepare_data

whole_data_path = '../../../data/digitoday/digitoday.2014.txt'
train_data_path = '../../../data/digitoday/digitoday.2014.train.txt'
dev_data_path = '../../../data/digitoday/digitoday.2014.dev.txt'
test_data_path = '../../../data/digitoday/digitoday.2015.test.txt'
wiki_data_path = '../../../data/digitoday/wikipedia.test.txt'


whole_data = prepare_data.load_whole_data(train_data_path, dev_data_path, test_data_path, wiki_data_path)
train_data = prepare_data.load_data(train_data_path)
dev_data = prepare_data.load_data(dev_data_path)
test_data = prepare_data.load_data(test_data_path)
wiki_data = prepare_data.load_data(wiki_data_path)


def get_vocab(data):
    vocab_list = []
    for seq in data:
        for word in seq[0]:
            vocab_list.append(word)
    return vocab_list

whole_vocab_list = get_vocab(whole_data)
train_vocab_list = get_vocab(train_data)
dev_vocab_list = get_vocab(dev_data)
test_vocab_list = get_vocab(test_data)
wiki_vocab_list = get_vocab(wiki_data)


def write_to_file(file_path, vocab_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in vocab_list:
            f.write(word + '\n')

write_to_file('../output/vocab/whole_vocab.txt', whole_vocab_list)
write_to_file('../output/vocab/train_vocab.txt', train_vocab_list)
write_to_file('../output/vocab/dev_vocab.txt', dev_vocab_list)
write_to_file('../output/vocab/test_vocab.txt', test_vocab_list)
write_to_file('../output/vocab/wiki_vocab.txt', wiki_vocab_list)