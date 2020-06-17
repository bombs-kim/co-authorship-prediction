from itertools import combinations
from random import shuffle, sample

import numpy as np
import torch
from torch.utils.data import Dataset


class FixedLengthContextDataset(Dataset):
    def __init__(self, path, max_context=3, negative_sample=10):
        self.tups = []
        self.max_context = max_context
        self.negative_sample = negative_sample
        with open(path) as f:
            it = iter(f)
            num_authors, num_papers = (int(n) for n  in next(f).split())
            self.num_authors = int(num_authors)

            for idx, line in enumerate(it):
                coauthors = tuple(int(n) for n in line.split())
                self.tups.append(coauthors)

            if num_papers != idx+1:
                print("num papers does not match", idx+1, num_papers)

        self.authors = tuple(range(1, self.num_authors+1))

    def __len__(self):
        return len(self.tups)

    def __getitem__(self, idx):
        l = list(self.tups[idx])
        shuffle(l)

        input_author = [l.pop()]
        if len(l) >= self.max_context:
            context = sample(l, self.max_context)
        else:
            while len(l) != self.max_context:
                l = (l + l)[:self.max_context]
            context = l

        # Negative sample
        # To understanding, negative_sample * max_context should be used!
        neg = sample(self.authors, self.negative_sample * self.max_context)

        ret = input_author, context, neg
        # Change 1-based index to 0-based index
        return tuple(torch.tensor(t) - 1 for t in (ret))



class HyperedgeDataset(Dataset):
    def __init__(self, filepath='./paper_author.txt', zero_based=True):
        super(HyperedgeDataset).__init__()
        self.filepath = filepath

        f = open(filepath, 'r')
        lines = f.readlines()
        self.nodes = set()
        self.hyperedges = []

        idx_correction = 1 if zero_based else 0

        for line in lines[1:]:
            node = line.strip().split(' ')
            # IMPORTANT: 0-based index may be used
            node = [int(i) - idx_correction for i in node]
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
    def __init__(self, split='train', ratio=0.8,
                 querypath='./data/query_public.txt', answerpath='./data/answer_public.txt',
                 zero_based=True):
        super(QueryDataset, self).__init__()

        query_public = open(querypath, 'r')
        answer_public = open(answerpath, 'r')
        query_lines = query_public.readlines()
        answer_lines = answer_public.readlines()

        self.zero_based = zero_based
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
            print('training classifier with {:d} data'.format(len(self.collaborations)))
        elif self.split == 'valid':
            self.collaborations = np.array(self.collaborations)[val_idx]
            self.labels = np.array(self.labels)[val_idx]
            print('validate classifier with {:d} data'.format(len(self.collaborations)))

    def __len__(self):
        return len(self.collaborations)

    def __getitem__(self, idx):
        authors = torch.tensor(self.collaborations[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        idx_correction = 1 if self.zero_based else 0
        # IMPORTANT: 0-based index may be used
        return authors - idx_correction, label
