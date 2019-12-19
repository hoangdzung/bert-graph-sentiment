import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm 

MAX_LEN = 96

def get_split_dataloader(sentences, labels, tokenizer, batch_size):
    input_ids = []

    for sent in tqdm(sentences):
        encoded_sent = tokenizer.encode(sent,add_special_tokens = True)
        input_ids.append(encoded_sent)

    inputs = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                                    value=0, truncating="post", padding="post")                                                     
    masks = []

    for sent in inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        masks.append(att_mask)

    inputs = torch.tensor(inputs)
    labels = torch.tensor(labels)
    masks = torch.tensor(masks)

    data = TensorDataset(inputs, masks, labels)
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

   return dataloader

def get_baseline_dataloader(datafile, batch_size, tokenizer):
    df = pd.read_pickle(datafile)
    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    train_sentences = df[df['split']=='train'].sentence.values
    dev_sentences = df[df['split']=='dev'].sentence.values
    test_sentences = df[df['split']=='test'].sentence.values

    train_labels = df[df['split']=='train'].label.values.astype(int)
    validation_labels = df[df['split']=='dev'].label.values.astype(int)
    test_labels = df[df['split']=='test'].label.values.astype(int)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_dataloader = get_split_dataloader(train_sentences, train_labels, tokenizer, batch_size)
    validation_dataloader = get_split_dataloader(dev_sentences, validation_labels, tokenizer, batch_size)
    test_dataloader = get_split_dataloader(test_sentences, test_labels, tokenizer, batch_size)

    return train_dataloader, validation_dataloader, test_dataloader