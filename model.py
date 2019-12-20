import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

class RGCNLayer(nn.Module):
    def __init__(self, feat_size, out_size, num_rels, activation=None, gated = True):
        
        super(RGCNLayer, self).__init__()
        self.feat_size = feat_size
        self.num_rels = num_rels
        self.activation = activation
        self.gated = gated

        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, out_size))
        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,gain=nn.init.calculate_gain('relu'))
        
        if self.gated:
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.feat_size, 1))
            nn.init.xavier_uniform_(self.gate_weight,gain=nn.init.calculate_gain('sigmoid'))
        
    def forward(self, g):
        
        weight = self.weight
        gate_weight = self.gate_weight
        
        def message_func(edges):
            w = weight[edges.data['rel_type']]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm'].unsqueeze(-1)
            if self.gated:
                gate_w = gate_weight[edges.data['rel_type']]
                gate = torch.bmm(edges.src['h'].unsqueeze(1), gate_w).squeeze().reshape(-1,1)
                gate = torch.sigmoid(gate)
                msg = msg * gate
                
            return {'msg': msg}
    
        def apply_func(nodes):
            h = nodes.data['h']
            h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)
        return g

class RGCNModel(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_rels, gated = True):
        super(RGCNModel, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.num_rels = num_rels
        self.gated = gated
        
        # create rgcn layers
        self.build_model()
       
    def build_model(self):        
        self.layers = nn.ModuleList() 
        self.layers.append(RGCNLayer(self.in_size, self.hidden_size, self.num_rels, activation=F.relu, gated = self.gated))
        self.layers.append(RGCNLayer(self.hidden_size, self.out_size, self.num_rels, activation=F.relu, gated = self.gated))
        
    
    def forward(self, g):
        for layer in self.layers:
            g = layer(g)
        
        rst_hidden = []
        for sub_g in dgl.unbatch(g):
            rst_hidden.append(  torch.mean(sub_g.ndata['h'], dim=0, keepdim=True)   )
        return  torch.cat(rst_hidden,dim=0)

class BERT_RGCN(nn.Module):
    """The main model."""
    def __init__(self, hidden_size, out_size, n_classes, bert_model):
        super().__init__()
        self.RGCN =  RGCNModel(in_size=768, hidden_size=hidden_size, out_size=out_size, num_rels = 3, gated = True)
        self.BERThead = bert_model # bert output size
        self.head = nn.Linear(768+out_size,n_classes)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, g, token_ids, masks, sent_len, labels):
        features_g, out_bert = self.BERThead(token_ids, attention_mask=masks)
        feats = []
        for i in range(token_ids.shape[0]):
            feats.append(features_g[i][1:1+sent_len[i]])

        feats = torch.cat(feats,dim=0)
        g.ndata['h'] = feats
        out_rgcn = self.RGCN(g)
        combine_out = torch.cat([out_bert, out_rgcn],dim=1)
        final_out = self.head(combine_out)
        #if self.training:
        return self.criterion(final_out, labels), final_out
        #else:
        #    return final_out
