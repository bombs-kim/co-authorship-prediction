"""
Note:
  Test accuracy with the best model

Usage:
  eval.py (--classifier <str>) (--embedding <str>) (--lstm | --deepset) [options]
  eval.py (-h | --help)

Options:
  --classifier <str>    Path for trained classifier.pth (required)
  --embedding <str>     Path for embedding.pth (required)
  --answer-path <str>   File name to save answer [default: ./data/answer_private.txt]
  --query-path <str>    File name to draw query [default: ./data/query_private.txt]
  --device <int>        Cuda device         [default: 0]

  --lstm                When set, use bidirectional LSTM aggregator
  --deepset             When set, use DeepSet aggregator
  --hidden <int>        Hidden size         [default: 256]
  --dropout <float>     Dropout rate        [default: 0.5]
  --enable-all-pools    (DeepSet only option) enable all poolings

  -h --help             Show this screen
"""

import os

# parsing library for cmd options and arguments https://github.com/docopt/docopt
from docopt import docopt
import torch
from torch.utils.data import DataLoader

from data import QueryDataset
from model import Classifier
from utils import load_embedding


def test_classifier(valid_loader, classifier, device, threshold):
    correct = 0
    positive = 0
    true_positive = 0
    false_negative = 0

    for collabs, label in valid_loader:
        score = classifier(collabs.to(device))
        prediction = score.cpu() >= threshold
        correct += int(prediction == label)
        positive += int(prediction)
        true_positive += int(prediction and label)
        false_negative += int(not prediction and label)
    try:
        precision =  true_positive / positive
        recall = true_positive / (true_positive + false_negative)
        acc = (correct / len(valid_loader))
        classifier.train()

        log_msg = f'Threshold: {threshold:.4f} | Val Acc: {acc:.4f} | '\
                f'Precision {precision:.4f} | Recall {recall:.4f}'
        print(log_msg)
    except:
        print('error', threshold)


def main():
    args = docopt(__doc__)
    enable_all_pools = args['--enable-all-pools']

    hidden = int(args['--hidden'])
    dropout = float(args['--dropout'])
    device = torch.device(int(args['--device']))
    print(f"{device} will be used")
    ratio  = 0.8

    valid_dset = QueryDataset(split='valid', ratio=ratio, equally_handle_foreign_authors=False)
    valid_loader = DataLoader(valid_dset, batch_size=1, num_workers=1, shuffle=False)

    embedding_mode, embedding = load_embedding(
        args['--embedding'], False, device)
    classifier = Classifier(
        embedding, hidden, dropout, args['--deepset'],
        equally_handle_foreign_authors=False, enable_all_pools=enable_all_pools)
    classifier.load_state_dict(torch.load(args['--classifier']))
    classifier.eval()

    if torch.cuda.is_available():
        classifier.to(device)

    thresholds = [0.05 * i for i in range(1, 20)]
    for thres in thresholds:
        test_classifier(valid_loader, classifier, device, thres)


if __name__ == '__main__':
    main()

