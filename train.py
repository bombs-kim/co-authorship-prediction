import torch
import torch.nn as nn

from tqdm import tqdm
import time

def train_embedding(trainloader, model, optimizer, criterion, device, epoch, args):
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

        if (i+1) % args.batch_size == 0:
            loss /= args.batch_size
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
    print('\nEpoch {:d} | Avg Loss: {:.6f}'.format(epoch, avg_loss))

    return avg_loss

def train_classifier(trainloader, validloader, embedding_model, classifier, optimizer, device, epoch, args):
    pbar = tqdm(total=len(trainloader), initial=0, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    running_loss = 0
    avg_loss = 0
    count = 0
    loss = 0

    for i, (nodes, label) in enumerate(trainloader):
        nodes = nodes.squeeze().to(device)
        with torch.no_grad():
            feats = embedding_model(nodes)
        feats = feats.unsqueeze(1)
        score = classifier(feats)

        step_loss = torch.abs(label.to(device) - score)
        loss += step_loss
        avg_loss += step_loss.item()
        running_loss += step_loss.item()

        if (i+1) % args.batch_size == 0:
            loss /= args.batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = 0

        count += 1
        if (i+1) % 1000 == 0:
            correct = 0
            classifier.eval()
            with torch.no_grad():
                for nodes, label in validloader:
                    nodes = nodes.squeeze().to(device)
                    feats = embedding_model(nodes)
                    feats = feats.unsqueeze(1)
                    score = classifier(feats).round()
                    correct += (score.cpu() == label).float().item()
            
            acc = correct / len(validloader)
            classifier.train()    
            pbar.set_description('train_loss: {:.6f}, valid_acc: {:.2f}%'.format(running_loss/count, acc*100))
            running_loss = 0
            count = 0
        
        pbar.update(1)

    avg_loss /= len(trainloader)
    last_acc = acc
    print('\nEpoch {:d} | Avg Loss: {:.6f} | Val Acc: {:.2f}%'.format(epoch, avg_loss, last_acc*100))

    return avg_loss, last_acc

if __name__ == '__main__':
    device = torch.device('cuda')

    embedding_model = Embedding().to(device)
    optimizer = optim.Adam(embedding_model.parameters(), lr=1e-3)
    criterion = CosineLoss().to(device)

    trainset = HyperedgeDataset(hyperedges)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)

    train_embedding(trainloader, embedding_model, optimizer, device)

    #after training embedding model
    validset = HyperedgeDataset(collaborations, labels)
    validloader = DataLoader(validset, batch_size=1, shuffle=True)
    
    model = Embedding().to(device)
    model.load_state_dict(torch.load('./embedding.pth'))
    classifier = Classifier().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    train_classifier(validloader, model, classifier, optimizer, device) 
