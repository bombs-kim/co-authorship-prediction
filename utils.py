from collections import OrderedDict
import datetime
import os

import torch
import torch.nn as nn
import random

from torch.utils.data import Dataset
import numpy as np

KST = datetime.timezone(datetime.timedelta(hours=9))

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


def now_kst():
    return datetime.datetime.now(tz=KST).strftime('%H:%M')


def get_dirname(mode, is_embedding=True):
    t = datetime.datetime.now(tz=KST).strftime('%m%d_%H%M')
    if is_embedding:
        dname = f'./backup/embedding_{mode}_{t}'
    else:
        dname = f'./backup/classifier_{mode}_{t}'

    if not os.path.exists(dname):
        os.makedirs(dname)
    return dname


def load_embedding(embedding_path, requires_grad=True):
    state = torch.load(embedding_path)

    if 'u_embedding.weight' in state:
        weight = state['u_embedding.weight']
        state = OrderedDict()
        state['weight'] = weight

    vocabulary_size, embedding_dim = state['weight'].shape
    model = nn.Embedding(vocabulary_size, embedding_dim, sparse=True)
    model.load_state_dict(state)
    model.requires_grad_(requires_grad)  # May not be needed
    return model, embedding_dim


