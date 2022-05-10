#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
import numpy as np
import string
from time import time
from joblib import Memory
import os
from tqdm import tqdm
from datetime import datetime
import scipy.sparse as sp
memory = Memory(location='.cache_data', verbose=0)


# In[ ]:


UNKNOWN_TOKEN = '<UNK>'
PADDING_TOKEN = '<PAD>'


# In[ ]:


class Arguments:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    embedding_size = 300
    corpus_size = 9999
    batch_size = 16
    accumulate = 2
    num_workers = 8
    n = 2
    max_length = 5000
    dropout = 0.5
    lr = 1e-3
    weight_decay = 1e-4
    epoch = 10
    prefetch_factor = 10
    hidden = 8
    nb_heads = 8
    alpha = 0.2


# In[ ]:


def _load_Glove(args):
    stoi = {}
    itos = {}
    embedding = []
    # Add unknown and padding token into the embedding
    stoi[UNKNOWN_TOKEN] = 0
    itos[0] = UNKNOWN_TOKEN
    embedding.append(np.random.rand(args.embedding_size))
    stoi[PADDING_TOKEN] = 1
    itos[1] = PADDING_TOKEN
    embedding.append(np.random.rand(args.embedding_size))
    
    with open('embeddings/glove.6B.{}d.txt'.format(args.embedding_size), 'r', encoding='utf8') as f:
        for idx, line in enumerate(f, start=2):
            values = line.split()
            stoi[values[0]] = idx
            itos[idx] = values[0]
            embedding.append([float(v) for v in values[1:]])
            if idx > args.corpus_size:
                break
        
        embedding = np.array(embedding, dtype=np.float32)
    return stoi, itos, embedding


# In[ ]:


class FakeStyleDataset(Dataset):
    def __init__(self, data, stoi, args):
        super(FakeStyleDataset).__init__()
        nodes = [[word for word in sentence.strip().split(' ')[:args.max_length]] for _, sentence in data]
        nodes = [[stoi[word] if word in stoi else stoi[UNKNOWN_TOKEN] for word in sentence] for sentence in nodes]
        self.nodes = nodes
        
        self.stoi = stoi
        self.n = args.n
        self.labels = [label for label, _ in data]
        
    def __getitem__(self, idx):
        neigh = []
        for j in range(len(self.nodes[idx])):
            n = []
            for i in range(-self.n, self.n + 1):
                if 0 <= i + j < len(self.nodes[idx]):
                    n.append(self.nodes[idx][i+j])
                else:
                    n.append(self.stoi[PADDING_TOKEN])
            neigh.append(n)

        return (
            torch.LongTensor(self.nodes[idx]),
            torch.LongTensor(neigh),
            self.labels[idx]
        )
    
    def __len__(self):
        return len(self.nodes)
    


# In[ ]:


def _load_data(scenario, fold):
    df_data = pd.read_csv('./corpusSources.tsv', sep='\t', encoding='utf-8')
    valid_data = ~df_data['content'].isna()
    df_data = df_data[valid_data][['Non-credible', 'content']]
    df_data['content'] = df_data['content'].str.lower()
    df_data['content'] = df_data['content'].str.replace('[{}]'.format(string.punctuation), ' ')
    df_data['content'] = df_data['content'].str.replace('\s+', ' ', regex=True)
    print('Finished pre-process DF data')

    # k-fold with k = 5
    df_fold = pd.read_csv('./foldsCV.tsv', sep='\t', encoding='utf-8')
    df_fold = df_fold[valid_data]
    fold_list = df_fold[scenario + 'CV'].to_numpy() ####################################################

    fold_idx = list(range(1, 6))
    fold_idx = fold_idx[-fold:] + fold_idx[:-fold] # Rotate the fold depending on fold input
    fold_idx = {
        'train': fold_idx[0:4],
        'val': fold_idx[4:5],
        'test': fold_idx[4:5]
    }

    included_data = np.isin(fold_list, fold_idx)

    train_data = df_data.to_numpy()[np.isin(fold_list, fold_idx['train'])]
    val_data = df_data.to_numpy()[np.isin(fold_list, fold_idx['val'])]
    test_data = df_data.to_numpy()[np.isin(fold_list, fold_idx['test'])]
    
    return train_data, val_data, test_data


# In[ ]:


def _load_dataset(train_data, val_data, stoi, args):
    train_dataset = FakeStyleDataset(train_data, stoi, args)
    val_dataset = FakeStyleDataset(val_data, stoi, args)
    
    return train_dataset, val_dataset


# In[ ]:


def _load_dataloader(train_dataset, val_dataset, stoi, args):
    
    def pad_collate(batch):
        node_list = []
        neighbor_list = []
        label_list = []
        for sample in batch:
            node_list.append(sample[0])
            neighbor_list.append(sample[1])
            label_list.append(sample[2])
        node_list = nn.utils.rnn.pad_sequence(node_list, batch_first=True, padding_value=stoi[PADDING_TOKEN])
        
        max_len_0 = max([s.shape[0] for s in neighbor_list])
        max_len_1 = max([s.shape[1] for s in neighbor_list])
        out_dims = (len(neighbor_list), max_len_0, max_len_1)
        out_tensor = neighbor_list[0].data.new(*out_dims).fill_(stoi[PADDING_TOKEN])
        for i, tensor in enumerate(neighbor_list):
            len_0 = tensor.size(0)
            len_1 = tensor.size(1)
            out_tensor[i, :len_0, :len_1] = tensor
        neighbor_list = out_tensor
        label_list = torch.LongTensor(label_list)        
        
        # Construct adjacency matrix
        adj_list = []
        for neighbors_batch in neighbor_list:
            edges = []
            for neighbors in neighbors_batch:
                for i in range(1,args.n+1):
                    edges.append([neighbors[args.n],neighbors[args.n+i]])
                    edges.append([neighbors[args.n],neighbors[args.n-i]])
            edges_tensor = torch.Tensor(edges)
            adj = sp.coo_matrix((np.ones(edges_tensor.shape[0]), (edges_tensor[:, 0], edges_tensor[:, 1])), shape=(args.corpus_size+2, args.corpus_size+2), dtype=np.float32)        
            adj_list.append(adj)
        
        return adj_list, node_list, neighbor_list, label_list
    
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True, prefetch_factor=args.prefetch_factor, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False, prefetch_factor=args.prefetch_factor, collate_fn=pad_collate)

    return train_loader, val_loader


# In[ ]:


class FakeStyleGraph(nn.Module):
    
    def __init__(self, embedding, args, stoi):
        super(FakeStyleGraph, self).__init__()
        
        adjacency_matrix = torch.Tensor(embedding.shape[0], embedding.shape[0])
        nn.init.xavier_uniform_(adjacency_matrix, gain=0.5) # To ensure the initialization is not too big
        self.adjacency_matrix = nn.Parameter(adjacency_matrix) # Smaller is better to prevent under-trained
        print("ADJ")
        print(self.adjacency_matrix)
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding), freeze=False)
        self.aggregate_rate = nn.Parameter(torch.rand(embedding.shape[0])) # To ensure the initialization is balanced enough
        
        self.dropout = nn.Dropout(args.dropout)
        self.last = nn.Linear(args.embedding_size, 2)
        
    def forward(self, nodes, neighbors, stoi, args):
        nodes_embedding = self.embedding(nodes)
        neighbors_embedding = self.embedding(neighbors)
        temp_nodes = nodes.unsqueeze(2).repeat(1, 1, neighbors.shape[-1])
        h = self.adjacency_matrix[temp_nodes, neighbors]
        # Disable weight that is just for padding
        h = h.masked_fill(temp_nodes == stoi[PADDING_TOKEN], 0)
        
        M = h.reshape(-1, 1) * neighbors_embedding.reshape(-1, args.embedding_size)
        M = self.dropout(M)
        M = M.reshape(neighbors_embedding.shape)
        M, _ = torch.max(M, dim=2)
        
        # Disable representations for padding
        message_aggregate_rate = self.aggregate_rate[nodes].masked_fill(nodes == stoi[PADDING_TOKEN], 0)
        ori_aggregate_rate = (1 - message_aggregate_rate).masked_fill(nodes == stoi[PADDING_TOKEN], 0)
        representations = message_aggregate_rate.unsqueeze(-1) * M + ori_aggregate_rate.unsqueeze(-1) * nodes_embedding
        representations = torch.sum(representations, dim=1)
        label = self.last(representations)
        return label

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        adjacency_matrix = torch.Tensor(embedding.shape[0], embedding.shape[0])
        nn.init.xavier_uniform_(adjacency_matrix, gain=0.5) # To ensure the initialization is not too big
        self.adjacency_matrix = nn.Parameter(adjacency_matrix) # Smaller is better to prevent under-trained
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding), freeze=False)
        self.aggregate_rate = nn.Parameter(torch.rand(embedding.shape[0])) # To ensure the initialization is balanced enough
        
        #self.dropout = nn.Dropout(args.dropout)
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
        
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# In[ ]:


def main(scenario, fold):
    args = Arguments()

    print('Start to load GloVE Embedding')
    stoi, itos, embedding = _load_Glove(args)

    print('Start to load data')
    train_data, val_data, test_data = _load_data(scenario, fold)
    train_dataset, val_dataset = _load_dataset(train_data, val_data, stoi, args)
    print('Training Dataset:', len(train_dataset))
    print('Evaluation Dataset:', len(val_dataset))

    train_loader, val_loader = _load_dataloader(train_dataset, val_dataset, stoi, args)

    model = FakeStyleGraph(embedding, args, stoi)
    # model = GAT(nfeat=embedding.shape[1],
                # nhid=args.hidden, 
                # nclass=2, 
                # dropout=args.dropout, 
                # nheads=args.nb_heads, 
                # alpha=args.alpha)
    for p in model.parameters():
        if len(p.shape) > 1:
            nn.init.xavier_uniform_(p, gain=nn.init.calculate_gain('relu'))
        else:
            nn.init.uniform_(p, 0.0, 0.1)
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.to(args.device)

    for epoch in range(args.epoch):
        model.train()
        train_loss = 0
        train_correct = 0

        print('Start Training')
        # Because our VRAM is not that big, we need to simulate larger batch size
        # Using gradients accumulation
        # But batch size too small will make computation too slow
        model.zero_grad()
        for idx, (adjs, nodes, neighbors, labels) in tqdm(enumerate(train_loader)):
            print("adjs")
            for adj in adjs:
                print(adj)
            print("nodes")
            print(nodes)
            print("neighbors")
            print(neighbors)
            adjs = adjs.to(args.device)
            nodes = nodes.to(args.device)
            neighbors = neighbors.to(args.device)
            labels = labels.to(args.device)

            out = model(nodes, neighbors, stoi, args)
            loss = loss_func(out, labels).to(args.device)
            loss = loss / args.accumulate

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if (idx+1) % args.accumulate == 0:
                optimizer.step()
                model.zero_grad()
            train_loss += loss.item()
            train_correct += (out.argmax(dim=1) == labels).sum().item()
            break ###
            if (idx+1) % (100*args.accumulate) == 0:
                print('Loss', loss)
        train_acc = train_correct / len(train_dataset)

        print('Start Evaluation')
        model.eval()
        val_loss = 0
        val_correct = 0
        for idx, (adjs, nodes, neighbors, labels) in tqdm(enumerate(val_loader)):
            nodes = nodes.to(args.device)
            neighbors = neighbors.to(args.device)
            labels = labels.to(args.device)

            out = model(nodes, neighbors, stoi, args)
            loss = loss_func(out, labels).to(args.device)
            val_loss += loss.item()
            val_correct += (out.argmax(dim=1) == labels).sum().item()
        val_acc = val_correct / len(val_dataset)

        print('{}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(epoch+1, train_loss, val_loss, train_acc, val_acc))


# In[ ]:


for scenario in ['source', 'document', 'topic']:
    for fold in [1, 2, 3, 4, 5]:
        main(scenario, fold)

