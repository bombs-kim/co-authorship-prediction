"""
Note:
  'classifier.pth' and 'log.txt' are saved under ./backup/.../
  directory. The argument parser of this script is automatically generated
  by docopt package using this help message itself.

Usage:
  train_classifier.py symmetric (--embedding <str>) [options]
  train_classifier.py skipgram  (--embedding <str>) [options]
  train_classifier.py (-h | --help)

Options:
  --embedding <str> Path for embedding.pth (required)
  --hidden <int>    Hidden size     [default: 128]
  --dropout <float> Dropout rate    [default: 0.2]
  -b --batch <int>  Batch size      [default: 100]
  --lr <float>      Learning rate   [default: 1e-3]
  -e --epochs <int> Epochs          [default: 100]
  -s --seed <int>   Random seed [default: 0]
  --device <int>    Cuda device [default: 0]
  --ratio <float>   Train validation split ratio [default: 0.2]
  -h --help         Show this screen
"""

import os

# parsing library for cmd options and arguments https://github.com/docopt/docopt
from docopt import docopt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import QueryDataset
from model import SkipGram, Classifier
from utils import get_dirname, now_kst


def load_embedding(args, requires_grad=True):
    state = torch.load(args['--embedding'])

    if args['symmetric']:
        vocabulary_size, embedding_dim = state['weight'].shape
        model = nn.Embedding(vocabulary_size, embedding_dim)
        model.load_state_dict(state)
        return model, embedding_dim

    if args['skipgram']:
        vocabulary_size, embedding_dim = state['u_embedding.weight'].shape
        model = SkipGram(vocabulary_size, embedding_dim)
        model.load_state_dict(state)
        embedding = model.u_embedding
        embedding.requires_grad_(requires_grad)
        return embedding, embedding_dim


def train_classifier(train_loader, valid_loader, embedding_model, classifier,
                     optimizer, device, epoch, batch_size, logdir=None):
    pbar = tqdm(total=len(train_loader), initial=0,
                bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    running_loss = 0
    avg_loss = 0
    count = 0
    loss = 0

    for i, (nodes, label) in enumerate(train_loader):
        nodes = nodes.squeeze().to(device)
        with torch.no_grad():
            feats = embedding_model(nodes)
        feats = feats.unsqueeze(1)
        score = classifier(feats)

        step_loss = torch.abs(label.to(device) - score)
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
            correct = 0
            classifier.eval()
            with torch.no_grad():
                for nodes, label in valid_loader:
                    nodes = nodes.squeeze().to(device)
                    feats = embedding_model(nodes)
                    feats = feats.unsqueeze(1)
                    score = classifier(feats).round()
                    correct += (score.cpu() == label).float().item()

            acc = (correct / len(valid_loader)) * 100
            classifier.train()
            pbar.set_description('train_loss: {:.6f}, valid_acc: {:.2f}%'.format(running_loss/count, acc))
            running_loss = 0
            count = 0

        pbar.update(1)

    avg_loss /= len(train_loader)
    last_acc = acc

    log_msg = f'Epoch {epoch:d} | Avg Loss: {avg_loss:.6f} | Val Acc: {last_acc:.2f}% | {now_kst()}'
    print(f'\n', log_msg)
    if logdir:
        path = os.path.join(logdir, 'log.txt')
        with open(path, 'a') as f:
            f.write(log_msg + '\n')

    return avg_loss, last_acc


def main():
    args = docopt(__doc__)
    np.random.seed(int(args['--seed']))
    hidden = int(args['--hidden'])
    droput = float(args['--dropout'])
    batch_size    = int(args['--batch'])
    lr     = float(args['--lr'])
    epochs = int(args['--epochs'])
    device = torch.device(int(args['--device']))
    ratio  = float(args['--ratio'])

    train_dset = QueryDataset(split='train')
    valid_dset = QueryDataset(split='valid')
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(valid_dset, batch_size=1, shuffle=False)

    embedding, embedding_dim = load_embedding(args)
    classifier = Classifier(embedding_dim, hidden, droput)
    if torch.cuda.is_available():
        embedding.to(device)
        classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    dname = get_dirname('symmetric', is_embedding=False)
    backup_path = os.path.join(dname, 'classifier.pth')

    best_acc = 0
    for epoch in range(epochs):
        avg_loss, val_acc = train_classifier(
            train_loader, valid_loader, embedding, classifier,
            optimizer, device, epoch, batch_size, dname)
        if val_acc > best_acc:
            torch.save(classifier.state_dict(), backup_path)

if __name__ == '__main__':
    main()