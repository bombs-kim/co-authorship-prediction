import torch
import torch.nn as nn
import random

from torch.utils.data import Dataset
import numpy as np


class Embedding(nn.Module):
    def __init__(self, N, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=N+1, embedding_dim=embedding_dim)
    
    def forward(self, x):
        # x should be LongTensor
        e = self.embedding(x)
        return e
    
    
class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, feats):
        loss = []
        num_nodes = feats.size(0)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                loss.append(self.cos(feats[i], feats[j]))

        loss = (1-torch.stack(loss)).mean()
        return loss


class HyperedgeDataset(Dataset):
    def __init__(self, filepath='./paper_author.txt'):
        super(HyperedgeDataset).__init__()
        self.filepath = filepath

        f = open(filepath, 'r')
        lines = f.readlines()
        self.nodes = set()
        self.hyperedges = []

        for line in lines[1:]:
            node = line.strip().split(' ')
            node = [int(i) for i in node]
            self.nodes.update(node)
            self.hyperedges.append(node)

        self.N = int(lines[0].split(' ')[0])
        self.M = int(lines[0].split(' ')[1])
        f.close()

    def __len__(self):
        return len(self.hyperedges)
    
    def __getitem__(self, idx):
        return torch.tensor(self.hyperedges[idx], dtype=torch.long)


class QueryDataset(Dataset):
    def __init__(self, split='train', ratio=0.8, querypath='query_public.txt', answerpath='answer_public.txt'):
        super(QueryDataset, self).__init__()

        query_public = open(querypath, 'r')
        answer_public = open(answerpath, 'r')
        query_lines = query_public.readlines()
        answer_lines = answer_public.readlines()
        self.nodes = set()
        self.collaborations = []
        self.labels = []
        self.split = split
        self.ratio = ratio

        for i, line in enumerate(query_lines[1:]):
            label = answer_lines[i].strip()
            if label == 'True':
                self.labels.append(1)
            elif label == 'False':
                 self.labels.append(0)
            else:
                raise NotImplementedError 
            node = line.strip().split(' ')
            node = [int(i) for i in node]
            self.nodes.update(node)
            self.collaborations.append(node)
        
        query_public.close()
        answer_public.close()

        perm = np.random.permutation(len(self.collaborations))
        split_idx = int(ratio*len(self.collaborations))
        train_idx = perm[:split_idx]
        val_idx = perm[split_idx:]
        if self.split == 'train':
            self.collaborations = np.array(self.collaborations)[train_idx]
            self.labels = np.array(self.labels)[train_idx]
        elif self.split == 'valid':
            self.collaborations = np.array(self.collaborations)[val_idx]
            self.labels = np.array(self.labels)[val_idx]
        
    def __len__(self):
        return len(self.collaborations)

    def __getitem__(self, idx):
        return torch.tensor(self.collaborations[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)
    

class Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_size, dropout_rate):
        super(Classifier, self).__init__()
        self.rnn = nn.LSTM(embedding_dim, hidden_size=hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        _, (hidden, cell) = self.rnn(x)
        hidden = hidden.squeeze()
        hidden = torch.cat([hidden[-2], hidden[-1]])
        out = self.dropout(hidden)
        out = self.fc1(out)
        out = self.relu(out)
        out = torch.sigmoid(self.fc2(out))

        return out
