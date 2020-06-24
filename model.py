import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MovingAverageNorm(nn.BatchNorm1d):
    """Code based on https://github.com/ptrblck/pytorch_misc"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True):
        super().__init__(
            num_features, eps, momentum, affine, True)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        # calculate running estimates
        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

            batch_mean = input.mean([0])
            # batch_var = input.var([0], unbiased=False)
            # TODO: make sure batch_var formula is okay
            batch_var = ((input - self.running_mean[None, :]) ** 2).mean([0])
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * batch_mean\
                    + (1 - exponential_average_factor) * self.running_mean
                self.running_var = exponential_average_factor * batch_var\
                    + (1 - exponential_average_factor) * self.running_var

        mean = self.running_mean
        var = self.running_var

        input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input


class SymmetricEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim

    def forward(self, pos_u, pos_v, neg_v):
        u = self.embedding(pos_u)
        num_pos = pos_v.size(1)
        num_neg = neg_v.size(1)

        pos_v = self.embedding(pos_v)
        score = (u * pos_v).sum(dim=-1)
        score = F.logsigmoid(score)
        loss1 = score.sum(dim=-1)

        neg_v = self.embedding(neg_v)
        neg_score = (u * neg_v).sum(dim=-1)
        neg_score = F.logsigmoid(-1 * neg_score)
        loss2 = neg_score.sum(dim=-1)

        loss = ((loss1 + loss2) / (num_pos + num_neg)).mean()
        return -1 * loss

    @property
    def input_embedding(self):
        return self.u_embedding.weight.data.cpu().numpy()


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

        loss = ((loss1 + loss2) / (num_pos + num_neg)).mean()
        return -1 * loss

    @property
    def input_embedding(self):
        return self.u_embedding.weight.data.cpu().numpy()


class DeepSet(nn.Module):
    def __init__(self, embedding_size, out_size, hidden_size=None,
            sumpool=True, maxpool=False, avgpool=False, dropout_rate=0):
        """
        Based on Deep Sets paper https://arxiv.org/abs/1703.06114.
        When multiple pools are enabled, their outputs are concatenated
        before applying affine operation.
        """
        super().__init__()

        num_pools = sum(int(p) for p in (sumpool, maxpool, avgpool))
        assert num_pools>=1
        hidden_size = hidden_size if hidden_size else out_size

        # TODO: do maxpooll based on the absolute values

        self.pools = {
            'sumpool': torch.sum if sumpool else None,
            'maxpool': torch.max if maxpool else None,
            'avgpool': torch.mean if avgpool else None}

        for name, pool_op in self.pools.items():
            if pool_op:
                affine = nn.Linear(embedding_size, hidden_size)
                setattr(self, f'affine_before_{name}', affine)
            setattr(self, name, pool_op)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.affine = nn.Linear(hidden_size * num_pools, out_size)
        self.relu = nn.ReLU()
        # self.norm1 = MovingAverageNorm(hidden_size * num_pools)
        # self.norm2 = MovingAverageNorm(out_size)

    @property
    def savename(self):
        poolmode = ','.join(name for name, op in self.pools.items() if op)
        return f'deepset-{poolmode}'

    def forward(self, feats):
        pool_outputs = []
        for name, pool_op in self.pools.items():
            if pool_op:
                affine = getattr(self, f'affine_before_{name}')
                out = affine(feats)
                out = pool_op(out, dim=1)
                if not isinstance(out, torch.Tensor):
                    out = out[0]
                pool_outputs.append(out)
        out = torch.cat(pool_outputs, dim=-1)
        out = self.relu(out)  # IMPORTANT: newly added!
        # out = self.norm1(out)

        out = self.dropout(out)
        out = self.affine(out)
        out = self.relu(out)
        # out = self.norm2(out)
        return out


class BidirectionalLSTM(nn.Module):
    def __init__(self, embedding_size, out_size,
                 hidden_size=None, dropout_rate=0):
        super().__init__()
        hidden_size = hidden_size if hidden_size else out_size
        self.lstm = nn.LSTM(
            embedding_size, hidden_size=hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.affine = nn.Linear(hidden_size*2, out_size)  # bidirectional LSTM
        self.relu = nn.ReLU()
        # self.norm1 = MovingAverageNorm(hidden_size*2)
        # self.norm2 = MovingAverageNorm(out_size)

    def forward(self, feats):
        feats = feats.permute(1, 0, 2)
        # feats's shape: (seqlen, batch, feats_size)
        _, (hidden, cell) = self.lstm(feats)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.reshape(hidden.shape[0], -1)
        # hidden = self.norm1(hidden)
        out = self.dropout(hidden)
        out = self.affine(out)
        out = self.relu(out)
        # out = self.norm2(out)
        return out


class Classifier(nn.Module):
    def __init__(self, embedding, aggregator_out_size,
                 dropout_rate=0, deepset=False,
                 equally_handle_foreign_authors=True,
                 enable_all_pools=False):
        super().__init__()
        self.embedding = embedding
        _, embedding_size = embedding.weight.shape
        aggr = DeepSet if deepset else BidirectionalLSTM
        kwargs = {'dropout_rate': dropout_rate}
        if enable_all_pools:
            kwargs['maxpool'] = kwargs['avgpool'] = True
        self.aggregator = aggr(
            embedding_size, aggregator_out_size, **kwargs)
        self.affine = nn.Linear(aggregator_out_size, 1)

        self.equally_handle_foreign_authors = equally_handle_foreign_authors
        if equally_handle_foreign_authors:
            # Mokey-patch Embedding module
            foreign_emb = nn.Embedding(1 , embedding_size)
            # no_grad is needed since torch.cat operation is
            # traced by default for backpropagation
            with torch.no_grad():
                new_weight = torch.cat([embedding.weight, foreign_emb.weight])
            _emb_requires_grad = embedding.weight.requires_grad
            embedding.weight = nn.Parameter(new_weight)

            embedding.num_embeddings += 1
            embedding.requires_grad_(_emb_requires_grad)

    @property
    def savename(self):
        if isinstance(self.aggregator, BidirectionalLSTM):
            return 'classifier-lstm'
        else:
            return f'classifier-{self.aggregator.savename}'

    def forward(self, collabs):
        feats = self.embedding(collabs)
        out = self.aggregator(feats)
        out = torch.sigmoid(self.affine(out))
        return out
