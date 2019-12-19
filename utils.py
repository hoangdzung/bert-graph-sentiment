import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm 
import numpy as np

def get_dataloader(datafile, batch_size, tokenizer):
    df = pd.read_pickle(datafile)
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    train_sentences = df[df['split']=='train'].sentence.values
    dev_sentences = df[df['split']=='dev'].sentence.values
    test_sentences = df[df['split']=='test'].sentence.values

    train_labels = df[df['split']=='train'].label.values.astype(int)
    validation_labels = df[df['split']=='dev'].label.values.astype(int)
    test_labels = df[df['split']=='test'].label.values.astype(int)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_input_ids = []
    dev_input_ids = []
    test_input_ids = []

    for sent in tqdm(train_sentences, desc='read train'):
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
        train_input_ids.append(encoded_sent)

    for sent in tqdm(dev_sentences, desc='read dev'):
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
        dev_input_ids.append(encoded_sent)

    for sent in tqdm(test_sentences, desc='read test'):
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
        test_input_ids.append(encoded_sent)

    MAX_LEN = 96

    train_inputs = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", 
                                    value=0, truncating="post", padding="post")
    validation_inputs = pad_sequences(dev_input_ids, maxlen=MAX_LEN, dtype="long", 
                                    value=0, truncating="post", padding="post")
    test_inputs = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", 
                                    value=0, truncating="post", padding="post")                                                            

    train_masks = []
    validation_masks = []
    test_masks = []

    for sent in train_inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        train_masks.append(att_mask)

    for sent in validation_inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        validation_masks.append(att_mask)

    for sent in test_inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        test_masks.append(att_mask)    

    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    test_inputs = torch.tensor(test_inputs)

    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    test_labels = torch.tensor(test_labels)

    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    test_masks = torch.tensor(test_masks)


    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create the DataLoader for our test set.
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_dataloader, validation_dataloader, test_dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def get_acc(model, validation_dataloader, device):
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy/nb_eval_steps
    return eval_accuracy
