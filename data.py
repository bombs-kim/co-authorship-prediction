from itertools import combinations
import os
from random import choice, shuffle, sample

import numpy as np
import torch
from torch.utils.data import Dataset


class FixedLengthContextDataset(Dataset):
    def __init__(self, path, max_context=3, negative_sample_factor=10, zero_based=True):
        self.collabs = []
        self.max_context = max_context
        self.negative_sample_factor = negative_sample_factor

        idx_correction = 1 if zero_based else 0

        with open(path) as f:
            num_authors, num_collabs = (int(n) for n  in next(f).split())
            self.num_authors = int(num_authors)

            for idx, line in enumerate(f):
                # IMPORTANT: idx_correction may make the indices zero-based
                collab = tuple(int(n) - idx_correction for n in line.split())
                self.collabs.append(collab)
            assert num_collabs == idx+1

        self.authors = tuple(range(self.num_authors))

    def __len__(self):
        return len(self.collabs)

    def __getitem__(self, idx):
        l = list(self.collabs[idx])
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
        neg = sample(self.authors, self.negative_sample_factor * self.max_context)

        ret = input_author, context, neg
        return tuple(torch.tensor(t) for t in (ret))



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


# TODO: Added use_paper_author option to train_classifier
class QueryDataset(Dataset):
    def __init__(self, split='train', ratio=0.8,
                 query_path='./data/query_public.txt',
                 answer_path='./data/answer_public.txt',
                 authors_path='./data/paper_author.txt',
                 permpath='./perm.txt',
                 zero_based=True,
                 equally_handle_foreign_authors=False,
                 use_paper_author=False,
                 oversample_false_collabs=False):
        super(QueryDataset, self).__init__()

        query_public = open(query_path, 'r')
        answer_public = open(answer_path, 'r')
        query_lines = query_public.readlines()
        answer_lines = answer_public.readlines()

        self.zero_based = zero_based
        self.collabs = []
        self.labels = []
        self.split = split
        self.ratio = ratio

        idx_correction = 1 if zero_based else 0

        for i, line in enumerate(query_lines[1:]):
            label = answer_lines[i].strip()
            if label == 'True':
                self.labels.append(1)
            elif label == 'False':
                 self.labels.append(0)
            else:
                raise NotImplementedError
            collab = line.strip().split(' ')
            # IMPORTANT: idx_correction may make the indices zero-based
            collab = tuple(int(i) - idx_correction for i in collab)
            self.collabs.append(collab)

        query_public.close()
        answer_public.close()

        with open(permpath) as f:
            perm = []
            for num in f:
                num = int(num.strip())
                perm.append(num)
        assert len(perm) == len(self.collabs)

        split_boundary = int(ratio * len(self.collabs))
        self.train_indices = train_indices = perm[:split_boundary]
        self.val_indices = val_indices = perm[split_boundary:]

        if self.split == 'train':
            self.collabs = np.array(self.collabs)[train_indices]
            self.labels = np.array(self.labels)[train_indices]
            print(f'training classifier with {len(self.collabs)} data')
        elif self.split == 'valid':
            self.collabs = np.array(self.collabs)[val_indices]
            self.labels = np.array(self.labels)[val_indices]
            print(f'validate classifier with {len(self.collabs)} data')

        self.equally_handle_foreign_authors = equally_handle_foreign_authors
        if self.equally_handle_foreign_authors:
            print('Equally handle foreign authors')
            with open(authors_path) as f:
                num_authors, num_collabs = (int(n) for n  in next(f).split())
                all_authors = set(range(num_authors))
                seen_authors = set()
                for idx, line in enumerate(f):
                    # IMPORTANT: idx_correction may make the indices zero-based
                    coauthors = (int(n) - idx_correction for n in line.split())
                    seen_authors.update(coauthors)
                assert num_collabs == idx + 1
            self.foreign_authors = all_authors.difference(seen_authors)
            self.foreign_author_idx = len(all_authors)  # == last_idx + 1

        if oversample_false_collabs:
            assert len(self.collabs) == len(self.labels)
            false_collabs = []
            for collab, label in zip(self.collabs, self.labels):
                if label == 0:
                    false_collabs.append(collab)

        if self.split == 'train' and use_paper_author:
            self.collabs = list(self.collabs)
            self.labels = list(self.labels)
            with open(authors_path) as f:
                num_authors, num_collabs = (int(n) for n  in next(f).split())
                for idx, line in enumerate(f):
                    collab = tuple(int(n) - idx_correction for n in line.split())
                    self.collabs.append(collab)
                    self.labels.append(1)
            if oversample_false_collabs:
                for _ in range(num_collabs):
                    self.collabs.append(choice(false_collabs))
                    self.labels.append(0)
            print(f"Use {authors_path} data as training {len(self.collabs)}")

    def handle_foreign(self, collab):
        foreign_exist = False
        for idx, author in enumerate(collab):
            if author.item() in self.foreign_authors:
                collab[idx] = self.foreign_author_idx
                foreign_exist = True
        # Remove redundant or do sorting
        if foreign_exist:
            collab = torch.unique(collab)
        return collab

    def __len__(self):
        return len(self.collabs)

    def __getitem__(self, idx):
        l = list(self.collabs[idx])
        if self.split == 'train':
            shuffle(l)
        collab = torch.tensor(l, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.equally_handle_foreign_authors:
            collab = self.handle_foreign(collab)
        return collab, label


class QueryTestset(Dataset):
    def __init__(self, query_path='./data/query_private.txt', zero_based=True):
        super(QueryTestset, self).__init__()
        query_private = open(query_path, 'r')
        query_lines = query_private.readlines()

        self.collabs = []
        idx_correction = 1 if zero_based else 0
        for i, line in enumerate(query_lines[1:]):
            collab = line.strip().split(' ')
            # IMPORTANT: idx_correction may make the indices zero-based
            collab = tuple(int(i) - idx_correction for i in collab)
            self.collabs.append(collab)

        query_private.close()

    def __len__(self):
        return len(self.collabs)

    def __getitem__(self, index):
        collab = torch.tensor(self.collabs[index], dtype=torch.long)
        return collab


# Test
if __name__ == '__main__':
    double_cnt = {}
    dset1 = QueryDataset(equally_handle_foreign_authors=False)
    dset2 = QueryDataset(equally_handle_foreign_authors=True)
    example_foreign_data = [62, 71, 92, 99, 122, 127, 149]
    for i in example_foreign_data:
        print("without handle foriegn", dset1[i][0])
        print("   with handle foriegn", dset2[i][0])
        print()
