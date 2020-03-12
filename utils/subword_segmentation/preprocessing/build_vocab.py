import sys
sys.path.insert(0,'../..')

import prepare_data


whole_data_path = '../../../data/combined/whole.txt'

# whole_data = prepare_data.load_whole_data(train_data_path, dev_data_path, test_data_path, wiki_data_path)
whole_data = prepare_data.load_data(whole_data_path)

# convert to lower case
whole_data = prepare_data.to_lower(whole_data)

# remove punctuation
whole_data = prepare_data.remove_punct(whole_data)


def get_vocab(data):
    vocab_list = []
    for seq in data:
        for word in seq[0]:
            vocab_list.append(word)
    return vocab_list

whole_vocab_list = get_vocab(whole_data)


def write_to_file(file_path, vocab_list):
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in vocab_list:
            f.write(word + '\n')

write_to_file('../output/vocab/whole_vocab1.txt', whole_vocab_list)