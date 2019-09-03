data_path = '../output/vocab/finnish_vocab.txt'
whole_data_path = '../output/vocab/whole_vocab.txt'

vocab = []

with open(whole_data_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    vocab.append(line.replace('\n', ''))

# remove digits
vocab = [word for word in vocab if not word.isdigit()]

with open('../output/vocab/preprocessed_finnish_vocab.txt', 'w') as f:
    for word in vocab:
        f.write(word)
        f.write('\n')