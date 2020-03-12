import torch
from utils.early_stopping import EarlyStopping


def train(model, word_num_layers, char_num_layers, morph_num_layers, num_epochs, pairs_batch_train, pairs_batch_dev, word_hidden_size, char_hidden_size, morph_hidden_size, batch_size, criterion, optimizer, patience, device):

    early_stopping = EarlyStopping(patience=patience, verbose=False, delta=0)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, batch in enumerate(pairs_batch_train):
            pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_char_seqs, char_seq_lengths, pad_morph_seqs, morph_seq_lengths = batch
            pad_input_seqs, pad_target_seqs, pad_char_seqs, pad_morph_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_char_seqs.to(device), pad_morph_seqs.to(device)
            model.zero_grad()

            word_hidden = model.init_hidden(word_num_layers, word_hidden_size, batch_size, device)
            char_hidden = model.init_hidden(char_num_layers, char_hidden_size, batch_size, device)
            morph_hidden = model.init_hidden(morph_num_layers, morph_hidden_size, batch_size, device)
            
            emissions = model(pad_input_seqs, input_seq_lengths, pad_char_seqs, char_seq_lengths, pad_morph_seqs, morph_seq_lengths, word_hidden, 
                                char_hidden, morph_hidden, batch_size)

            pad_target_seqs = pad_target_seqs.squeeze()
            
            mask = pad_target_seqs.clone()
            mask[mask != 0] = 1
            mask = mask.byte()

            loss = -model.crf(emissions, pad_target_seqs, mask=mask)
            loss.backward()
            train_loss += loss
            optimizer.step()
        
        # calculate validation loss
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for i, batch in enumerate(pairs_batch_dev):
                pad_input_seqs, input_seq_lengths, pad_target_seqs, target_seq_lengths, pad_char_seqs, char_seq_lengths, pad_morph_seqs, morph_seq_lengths = batch
                pad_input_seqs, pad_target_seqs, pad_char_seqs, pad_morph_seqs = pad_input_seqs.to(device), pad_target_seqs.to(device), pad_char_seqs.to(device), pad_morph_seqs.to(device)

                word_hidden = model.init_hidden(word_num_layers, word_hidden_size, batch_size, device)
                char_hidden = model.init_hidden(char_num_layers, char_hidden_size, batch_size, device)
                morph_hidden = model.init_hidden(morph_num_layers, morph_hidden_size, batch_size, device)
                
                emissions = model(pad_input_seqs, input_seq_lengths, pad_char_seqs, char_seq_lengths, pad_morph_seqs, morph_seq_lengths, word_hidden, 
                                char_hidden, morph_hidden, batch_size)

                                        
                pad_target_seqs = pad_target_seqs.squeeze()
                
                mask = pad_target_seqs.clone()
                mask[mask != 0] = 1
                mask = mask.byte()

                loss = -model.crf(emissions, pad_target_seqs, mask=mask)
                val_loss += loss


        # early_stopping(val_loss/len(pairs_batch_dev), model)
    
        # if early_stopping.early_stop:
            # print("Early stopping")
            # break
        
        # if epoch % 5 == 0:
        print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, train_loss/len(pairs_batch_train), val_loss/len(pairs_batch_dev)))

    print('\n The final loss is:')
    print('[Epoch: %d] train_loss: %.4f    val_loss: %.4f' % (epoch+1, train_loss/len(pairs_batch_train), val_loss/len(pairs_batch_dev)))
    torch.save(model.state_dict(), 'weights/model_upper.pt')
