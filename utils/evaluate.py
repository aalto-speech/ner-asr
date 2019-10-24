import torch
import numpy as np

import utils.prepare_data as prepare_data

def get_predictions(data, model, word_num_layers, char_num_layers, morph_num_layers, word_hidden_size, char_hidden_size, morph_hidden_size, batch_size, device):
    all_true = []
    predicted_tags = []
    for sent in range(len(data)):
        sentence = data[sent][0].to(device)
        tags = data[sent][1]
        chars = data[sent][2]
        morphs = data[sent][3]
        
        pad_char_seqs = prepare_data.pad_subwords(chars).to(device)
        pad_morph_seqs = prepare_data.pad_subwords(morphs).to(device)

        word_hidden = model.init_hidden(word_num_layers, word_hidden_size, batch_size, device)
        char_hidden = model.init_hidden(char_num_layers, char_hidden_size, batch_size, device)
        morph_hidden = model.init_hidden(morph_num_layers, morph_hidden_size, batch_size, device)

        emissions = model(sentence, [len(sentence)], pad_char_seqs, [pad_char_seqs.size(0)], pad_morph_seqs, [pad_morph_seqs.size(0)], word_hidden, char_hidden, morph_hidden, batch_size)
        
        mask = sentence.clone()
        mask[mask != 0] = 1
        mask = mask.byte()
        
        predicted_tags.append(model.crf.decode(emissions, mask=mask)[0])
        
        tags = tags.numpy().reshape(-1, )
        all_true.append(tags)
    
    predicted_tags = np.array(predicted_tags)
    return predicted_tags, all_true


def calculate_individual_score(all_predicted, all_true, b_tag_idx, i_tag_idx):
    precision = []
    recall = []
    tp = 0
    fp = 0
    fn = 0
    for seq in range(len(all_true)):
        for tag in range(len(all_true[seq])):
            if all_predicted[seq][tag] == b_tag_idx and all_true[seq][tag] == b_tag_idx:
                tp += 1
            elif all_predicted[seq][tag] == i_tag_idx and all_true[seq][tag] == i_tag_idx:
                tp += 1
            elif all_predicted[seq][tag] == b_tag_idx and all_true[seq][tag] != b_tag_idx:
                fp += 1
            elif all_predicted[seq][tag] == i_tag_idx and all_true[seq][tag] != i_tag_idx:
                fp += 1
            elif all_predicted[seq][tag] != b_tag_idx and all_true[seq][tag] == b_tag_idx:
                fn += 1
            elif all_predicted[seq][tag] != i_tag_idx and all_true[seq][tag] == i_tag_idx:
                fn += 1


    # calculate precision
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    # calculate recall
    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    
    precision = np.array(precision)
    recall = np.array(recall)

    # calculate f1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def calculate_average_score(all_predicted, all_true, b_tag_idx, i_tag_idx):
    tp = 0
    fp = 0
    fn = 0
    for seq in range(len(all_true)):      
        for tag in range(len(all_true[seq])):
            if all_predicted[seq][tag] == b_tag_idx and all_true[seq][tag] == b_tag_idx:
                tp += 1
            elif all_predicted[seq][tag] == i_tag_idx and all_true[seq][tag] == i_tag_idx:
                tp += 1
            elif all_predicted[seq][tag] == b_tag_idx and all_true[seq][tag] != b_tag_idx:
                fp += 1
            elif all_predicted[seq][tag] == i_tag_idx and all_true[seq][tag] != i_tag_idx:
                fp += 1
            elif all_predicted[seq][tag] != b_tag_idx and all_true[seq][tag] == b_tag_idx:
                fn += 1
            elif all_predicted[seq][tag] != i_tag_idx and all_true[seq][tag] == i_tag_idx:
                fn += 1
    
    return tp, fp, fn


def print_scores(all_predicted, all_true, tag2idx):
    org_precision, org_recall, org_f1 = calculate_individual_score(all_predicted, all_true, tag2idx['B-ORG'], tag2idx['I-ORG'])
    print('org precision: %.4f    org recall: %.4f    org_f1: %.4f' %(org_precision.mean(), org_recall.mean(), org_f1))

    per_precision, per_recall, per_f1 = calculate_individual_score(all_predicted, all_true, tag2idx['B-PER'], tag2idx['I-PER'])
    print('per precision: %.4f    per recall: %.4f    per_f1: %.4f' %(per_precision.mean(), per_recall.mean(), per_f1))

    loc_precision, loc_recall, loc_f1 = calculate_individual_score(all_predicted, all_true, tag2idx['B-LOC'], tag2idx['I-LOC'])
    print('loc precision: %.4f    loc recall: %.4f    loc_f1: %.4f' %(loc_precision.mean(), loc_recall.mean(), loc_f1))

    date_precision, date_recall, date_f1 = calculate_individual_score(all_predicted, all_true, tag2idx['B-DATE'], tag2idx['I-DATE'])
    print('date precision: %.4f    date recall: %.4f    date_f1: %.4f' %(date_precision.mean(), date_recall.mean(), date_f1))

    pro_precision, pro_recall, pro_f1 = calculate_individual_score(all_predicted, all_true, tag2idx['B-PRO'], tag2idx['I-PRO'])
    print('pro precision: %.4f    pro recall: %.4f    pro_f1: %.4f' %(pro_precision.mean(), pro_recall.mean(), pro_f1))

    event_precision, event_recall, event_f1 = calculate_individual_score(all_predicted, all_true, tag2idx['B-EVENT'], tag2idx['I-EVENT'])
    print('event precision: %.4f    event recall: %.4f    event_f1: %.4f' %(event_precision.mean(), event_recall.mean(), event_f1))

    org_tp, org_fp, org_fn = calculate_average_score(all_predicted, all_true, tag2idx['B-ORG'], tag2idx['I-ORG'])
    per_tp, per_fp, per_fn = calculate_average_score(all_predicted, all_true, tag2idx['B-PER'], tag2idx['I-PER'])
    loc_tp, loc_fp, loc_fn = calculate_average_score(all_predicted, all_true, tag2idx['B-LOC'], tag2idx['I-LOC'])
    date_tp, date_fp, date_fn = calculate_average_score(all_predicted, all_true, tag2idx['B-DATE'], tag2idx['I-DATE'])
    pro_tp, pro_fp, pro_fn = calculate_average_score(all_predicted, all_true, tag2idx['B-PRO'], tag2idx['I-PRO'])
    event_tp, event_fp, event_fn = calculate_average_score(all_predicted, all_true, tag2idx['B-EVENT'], tag2idx['I-EVENT'])

    micro_avg_precision = (org_tp + per_tp + loc_tp + date_tp + pro_tp + event_tp) / (org_tp + per_tp + loc_tp + date_tp + pro_tp + event_tp + org_fp + per_fp + loc_fp + date_fp + pro_fp + event_fp)
    micro_avg_recall = (org_tp + per_tp + loc_tp + date_tp + pro_tp + event_tp) / (org_tp + per_tp + loc_tp + date_tp + pro_tp + event_tp + org_fn + per_fn + loc_fn + date_fn + pro_fn + event_fn)
    avg_f1 = 2 * (micro_avg_precision * micro_avg_recall) / (micro_avg_precision + micro_avg_recall)
    print('micro avg precision: %.4f    micro avg recall: %.4f    avg f1: %.4f' %(micro_avg_precision, micro_avg_recall, avg_f1))


def evaluate_sentence(sentence_number, word_num_layers, char_num_layers, morph_num_layers, word_hidden_size, char_hidden_size, morph_hidden_size, batch_size, data, model, idx2word, idx2tag, device):
    test_sentence = data[sentence_number][0].to(device)

    chars = data[sentence_number][2]
    morphs = data[sentence_number][3]

    word_hidden = model.init_hidden(word_num_layers, word_hidden_size, batch_size, device)
    char_hidden = model.init_hidden(char_num_layers, char_hidden_size, batch_size, device)
    morph_hidden = model.init_hidden(morph_num_layers, morph_hidden_size, batch_size, device)

    pad_char_seqs = prepare_data.pad_subwords(chars).to(device)
    pad_morph_seqs = prepare_data.pad_subwords(morphs).to(device)

    emissions = model(test_sentence, [len(test_sentence)], pad_char_seqs.to(device), [pad_char_seqs.size(0)], pad_morph_seqs.to(device), [pad_morph_seqs.size(0)], word_hidden, char_hidden, morph_hidden, batch_size)

    print('\nPREDICTED TAGS: \n')

    for i in range(len(test_sentence)):
        word = idx2word[test_sentence[i].item()]
        tag = torch.argmax(emissions[i]).item()
        tag =  idx2tag[torch.argmax(emissions[i]).item()]
        
        print('{}: {}'.format(word, tag))

    print('\n\nREAL TAGS: \n')

    words = []
    tags = []
    for word in data[sentence_number][0]:
        words.append(idx2word[word.item()])
        
    for tag in data[sentence_number][1]:
        tags.append(idx2tag[tag.item()])    

    for i in range(len(words)):
        print(words[i] + ': ' + tags[i])


def evaluate_document(file, word_num_layers, char_num_layers, morph_num_layers, word_hidden_size, char_hidden_size, morph_hidden_size, batch_size, data, model, idx2word, idx2tag, device):
    with open (file, 'w', encoding='utf-8') as f:
        for sent in data:
            test_sentence = sent[0].to(device)

            chars = sent[2]
            morphs = sent[3]

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
                
                #  print('{}: {}'.format(word, tag))
                if word != '<start>' and word != '<end>':
                    f.write(word + '\t' + tag + '\n')
            f.write('\n')