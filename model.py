import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.u_embedding = nn.Embedding(
            vocab_size, embedding_dim, sparse=True)
        self.v_embedding = nn.Embedding(
            vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim

    def forward(self, pos_u, pos_v, neg_v):
        u = self.u_embedding(pos_u)
        num_pos = pos_v.size(1)
        num_neg = neg_v.size(1)

        v_embedding = self.v_embedding(pos_v)
        score = (u * v_embedding).sum(dim=-1)
        score = F.logsigmoid(score)
        loss1 = score.sum(dim=-1)

        neg_v = self.v_embedding(neg_v)
        neg_score = (u * neg_v).sum(dim=-1)
        neg_score = F.logsigmoid(-1 * neg_score)
        loss2 = neg_score.sum(dim=-1)

        loss = ((loss1 + loss2)/(num_pos + num_neg)).mean()

        return -1 * loss

    @property
    def input_embedding(self):
        return self.u_embedding.weight.data.cpu().numpy()


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
