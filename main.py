import torch
import torch.nn as nn
import logging
import os.path as osp
import time
import sys

from torch.utils.data import DataLoader

from dataset.SubsetDataset import SubsetDataset
from dataset.utils import GL_orig_RE
from evaluation.utils import Evaluator_DML
from net.load_net import load_net
from RAdam import RAdam

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logger = logging.getLogger('GNNReID.Training')


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Train Dataset
    dataset_name = 'CARS'
    root = f'./data/{dataset_name}'
    num_classes = 98
    included_labels = range(0, num_classes)
    labeled_fraction = 0.5
    
    transform = GL_orig_RE(is_train=True, RE=True)
    dataset = SubsetDataset(root, included_labels, labeled_fraction, transform)
    print('Dataset contains', len(dataset), 'samples')

    # Train DataLoader
    batch_size = 32
    num_workers = 1
    
    dl_tr = DataLoader(dataset,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=num_workers,
                       drop_last=True,
                       pin_memory=True)
    
    model, embed_size = load_net(num_classes=num_classes, pretrained_path='no', red=1)
    model = model.to(device)

    # Optimizer
    lr = 0.0001
    weight_decay = 0.000006
    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()
    evaluator = Evaluator_DML(dev=device)

    # Training
    epochs = 3
    start = time.time()
    for epoch in range(1, epochs + 1):
        logger.info(f'EPOCH {epoch}/{epochs}: {time.time() - start}')
        start = time.time()

        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            
            preds, fc7 = model(x)
            loss = loss_fn(preds, y)
            
            if torch.isnan(loss):
                print("We have NaN numbers, closing\n\n\n")
                return 0.0, model

            loss.backward()
            optimizer.step()
            break


    with torch.no_grad():
        logger.info('EVALUATION')
        filename = f'{dataset_name}_{time.time()}'
        mAP, top = evaluator.evaluate(model, dl_tr, dataroot='CARS', nb_classes=num_classes)         
        torch.save(model.state_dict(), osp.join('./results_nets', filename + '.pth'))


if __name__ == '__main__':
    main()