import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm 
import numpy as np
import dgl

MAX_LEN = 96

def sent2graph(sent, tokenizer):
    token_sent = tokenizer.tokenize(sent)
    sent = ' '.join([re.sub("[#]","",token)   for token in token_sent ])
    doc = parser(sent)
    parse_rst = doc.to_json()
    node2id = dict()
    edges = []
    edge_type = []
    for i_word, word in enumerate(parse_rst['tokens']):
        if i_word not in node2id:
            node2id[i_word] = len(node2id) 
            edges.append( [i_word, i_word] )
            edge_type.append(0)
        if word['head'] not in node2id:
            node2id[word['head']] = len(node2id) 
            edges.append( [word['head'], word['head']] )
            edge_type.append(0)

        if word['dep'] != 'ROOT':
            edges.append( [node2id[word['head']], node2id[word['id']]] )
            edge_type.append(1)
            edges.append( [node2id[word['id']], node2id[word['head']]] )
            edge_type.append(2)
            
    G = dgl.DGLGraph()
    G.add_nodes(len(node2id))
    G.add_edges(list(zip(*edges))[0],list(zip(*edges))[1]) 
    edge_norm = []
    for e1, e2 in edges:
        if e1 == e2:
            edge_norm.append(1)
        else:
            edge_norm.append( 1 / (G.in_degree(e2) - 1 ) )


    edge_type = torch.from_numpy(np.array(edge_type))
    edge_norm = torch.from_numpy(np.array(edge_norm)).float()

    G.edata.update({'rel_type': edge_type,})
    G.edata.update({'norm': edge_norm})
    
    return G, [101]+tokenizer.convert_tokens_to_ids(token_sent)+[102]

class GraphDataset(Dataset):
    def __init__(self, graphs, token_ids, masks, labels):

        self.graphs = graphs
        self.token_ids = token_ids  
        self.labels = labels  
        
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.token_ids[idx], self.labels[idx]

def collate(samples):
    graphs, token_ids, masks,labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    sent_len = [graph.number_of_nodes() for graph in graphs] 
    return batched_graph, torch.tensor(token_ids), torch.tensor(masks), torch.tensor(sent_len), torch.tensor(labels)

def get_split_dataloader(sentences, labels, tokenizer, batch_size):
    graphs = []
    token_ids = []
    for sent in tqdm(sentences, desc='gen graph'):
        graph, token_id = sent2graph(sent, tokenizer)
        graphs.append(graph)
        token_ids.append(token_id)

    token_ids = pad_sequences(token_ids, maxlen=MAX_LEN, dtype="long", 
                                    value=0, truncating="post", padding="post") 
    masks = []
    for token_id in token_ids:
        att_mask = [int(token > 0) for token in token_id]
        masks.append(att_mask)

    dataset = GraphDataset(graphs, token_ids, masks, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    return dataloader

def get_bert_rgcn_dataloader(datafile, batch_size, tokenizer):
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
