import torch
import torch.nn as nn
import random

from torch.utils.data import Dataset
import numpy as np


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
