import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from train import train_embedding, train_classifier
from utils import *
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--embed', action='store_true', default=False)
parser.add_argument('--classify', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--num-authors', type=int, default=58646)
parser.add_argument('--embedding-dim', type=int, default=128)
parser.add_argument('--hidden-size', type=int, default=128)
parser.add_argument('--dropout-rate', type=float, default=0.2)
parser.add_argument('--embed-lr', type=float, default=1e-1)
parser.add_argument('--classifier-lr', type=float, default=1e-2)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--ratio', type=float, default=0.8)
parser.add_argument('--embedding-path', type=str, default='./embedding.pth')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
device = torch.device('cuda')
np.random.seed(args.seed)

# training embedding model
if args.embed:
    trainset = HyperedgeDataset()
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)

    embedding_model = Embedding(trainset.N, args.embedding_dim).to(device)
    optimizer = optim.Adam(embedding_model.parameters(), lr=args.embed_lr)
    criterion = CosineLoss().to(device)
    
    min_loss = 999999
    for epoch in range(args.epochs):
        avg_loss = train_embedding(trainloader, embedding_model, optimizer, criterion, device, epoch, args)
        if avg_loss < min_loss:
            torch.save(embedding_model.state_dict(), args.embedding_path)

#after training embedding model
if args.classify:
    trainset = QueryDataset(split='train', ratio=args.ratio)
    validset = QueryDataset(split='valid', ratio=args.ratio)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    validloader = DataLoader(validset, batch_size=1, shuffle=False)

    model = Embedding(args.num_authors, args.embedding_dim).to(device)
    model.load_state_dict(torch.load(args.embedding_path))
    classifier = Classifier(args.embedding_dim, args.hidden_size, args.dropout_rate).to(device)
    optimizer_c = optim.Adam(classifier.parameters(), lr=args.classifier_lr)

    best_acc = 0
    for epoch in range(args.epochs):
        avg_loss, val_acc = train_classifier(trainloader, validloader, model, classifier, optimizer_c, device, epoch, args)
        if val_acc > best_acc:
            torch.save(classifier.state_dict(), './classifier.pth')

if args.test:
    testset = QueryDataset(split='valid', querypath='query_private.txt')
    test() 
