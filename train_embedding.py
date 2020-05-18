"""
Note:
  'embedding.pth' and 'log.txt' are saved under ./backup/.../
  directory. The argument parser of this script is automatically generated
  by docopt package using this help message itself.

Usage:
  train_embedding.py symmetric [options]
  train_embedding.py skipgram  [options]
  train_embedding.py (-h | --help)

Options:
  -d --dim <int>    Embedding dimension [default: 128]
  -b --batch <int>  Batch size          [default: 100]
  --lr <float>      Learning rate       [default: 1e-1]
  -e --epochs <int> Epochs              [default: 100]
  -s --seed <int>   Random seed [default: 0]
  --device <int>    Cuda device [default: 0]
  --file <str>      Path to input file [default: data/paper_author.txt]
  -h --help         Show this screen.
"""

import os
import time

# parsing library for cmd options and arguments https://github.com/docopt/docopt
from docopt import docopt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data import FixedLengthContextDataset, HyperedgeDataset
from model import SkipGram
from utils import CosineLoss, now_kst, get_dirname


def train_embedding(trainloader, model, optimizer, criterion, device, epoch, batch_size, logdir=None):
    pbar = tqdm(total=len(trainloader), initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    avg_loss = 0
    running_loss = 0
    count = 0
    loss = 0

    for i, edges in enumerate(trainloader):
        edges = edges.squeeze().to(device)
        feats = model(edges)

        step_loss = criterion(feats)
        loss += step_loss
        avg_loss += step_loss.item()
        running_loss += step_loss.item()

        if (i+1) % batch_size == 0:
            loss /= batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = 0

        count += 1
        if (i+1) % 1000 == 0:
            pbar.set_description('loss: {:.6f}'.format(running_loss/count))
            running_loss = 0
            count = 0
        pbar.update(1)

    avg_loss /= len(trainloader)

    log_msg = f'Epoch {epoch:d} | Avg Loss: {avg_loss:.6f} | {now_kst()}'
    print(f'\n', log_msg)
    if logdir:
        path = os.path.join(logdir, 'log.txt')
        with open(path, 'a') as f:
            f.write(log_msg + '\n')

    return avg_loss


def train(model, loader, epoch_num=100, lr=0.2, print_backup_interval=1000, device=None):
    print("start training")
    optimizer = optim.SGD(model.parameters(), lr=lr)
    dname = get_dirname('skipgram')
    recent_loss = 0

    for epoch in range(epoch_num):
        start = time.time()

        for batch_idx, (pos_u, pos_v, neg_v) in enumerate(loader):
            if torch.cuda.is_available():
                pos_u = pos_u.cuda(device=device)
                pos_v = pos_v.cuda(device=device)
                neg_v = neg_v.cuda(device=device)

            optimizer.zero_grad()
            loss = model(pos_u, pos_v, neg_v)

            loss.backward()
            optimizer.step()

            recent_loss = 0.95 * recent_loss + 0.05 * loss.item()

            if batch_idx % print_backup_interval == 0:
                join = os.path.join
                torch.save(model.state_dict(), join(dname, 'embedding.pth'))
                if epoch % 1000 == 0 or (epoch in (50, 100, 200, 400, 800)):
                    torch.save(model.state_dict(), join(dname, f"embedding_{epoch:05d}.pth"))

                batch_sec = print_backup_interval / (time.time() - start)
                log_msg = (f'epoch {epoch:2d} batch {batch_idx:5d}: '
                           f'loss {loss.item():6.4f}/{recent_loss:6.4f} {now_kst()}')
                print(log_msg, '\r', end='')
                with open(os.path.join(dname, "log.txt"), 'a') as f:
                    f.write(log_msg + '\n')
                start = time.time()
        print()


def main():
    args = args = docopt(__doc__)
    np.random.seed(int(args['--seed']))
    embedding_dim = int(args['--dim'])
    batch_size    = int(args['--batch'])
    lr     = float(args['--lr'])
    epochs = int(args['--epochs'])
    device = torch.device(int(args['--device']))
    fpath  = args['--file']

    # Symmetric vectors are used to compute cosine similarity
    if args['symmetric']:
        trainset = HyperedgeDataset(fpath)
        trainloader = DataLoader(trainset, batch_size=1, shuffle=True)

        embedding_model = nn.Embedding(trainset.N, embedding_dim).to(device)
        optimizer = optim.Adam(embedding_model.parameters(), lr=lr)
        criterion = CosineLoss().to(device)

        dname = get_dirname('symmetric')
        backup_path = os.path.join(dname, 'embedding.pth')

        min_loss = 999999
        for epoch in range(epochs):
            avg_loss = train_embedding(
                trainloader, embedding_model, optimizer, criterion,
                device, epoch, batch_size, dname)
            if avg_loss < min_loss:
                torch.save(embedding_model.state_dict(), backup_path)

    # Word2Vec Skip-gram. Unsymmetric vectors are used to compute cosine similarity
    elif args['skipgram']:
        max_context = 3
        neg_sample_num = 3
        dset = FixedLengthContextDataset(fpath, max_context, neg_sample_num)

        vocabulary_size = dset.num_authors
        model = SkipGram(vocabulary_size, embedding_dim)
        if torch.cuda.is_available():
            model.cuda(device=device)

        # TODO: Make num_workers cmd argument
        num_workers = 4
        loader = DataLoader(dset, batch_size, num_workers=num_workers)

        train(model, loader, epoch_num=epochs, lr=lr, device=device)


if __name__ == '__main__':
    main()