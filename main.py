import torch
import torch.nn as nn
import logging
import os.path as osp
import os
import time
import warnings
import argparse
import yaml

from torch.utils.data import DataLoader
from datetime import datetime

from dataset.SubsetDataset import SubsetDataset
from dataset.utils import GL_orig_RE
from evaluation.utils import Evaluator_DML
from net.load_net import load_net
from RAdam import RAdam

warnings.filterwarnings("ignore")


def main(config_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if not osp.isdir('./results_nets'):
        os.makedirs('./results_nets')
    if not osp.isdir('./results'):
        os.makedirs('./results')

    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s:%(name)s: %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    ### PARAMS ###
    # Dataset
    # root = f'/gdrive/MyDrive/DML/CARS'
    root = config['dataset']['path']
    dataset_name = config['dataset']['name']
    train_classes = config['dataset']['train_classes']
    labeled_fraction = config['dataset']['labeled_fraction']
    
    # DataLoader
    batch_size = 32
    num_workers = 4

    # Optimizer
    lr = 0.0001
    weight_decay = 0.000006

    # Training
    epochs = config['training']['epochs']
    ##############

    dl_tr, dl_ev = get_dataloaders(
        root,
        train_classes,
        labeled_fraction,
        batch_size,
        num_workers
    )
    
    model, embed_size = load_net(num_classes=train_classes, pretrained_path='no', red=4) # 4=512, 8=256
    model = model.to(device)
    print('Embedding size:', embed_size)

    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    evaluator = Evaluator_DML(config, device)

    scores = []
    best_recall_at_1 = 0
    best_filename = ''

    logger.info('TRAINING WITH {}% OF DATA'.format(labeled_fraction * 100))
    for epoch in range(1, epochs + 1):
        evaluator.logger.info('EPOCH {}/{}'.format(epoch, epochs))
        start = time.time()

        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            
            preds, embeddings = model(x, output_option='plain')
            loss = loss_fn(preds, y)
            
            if torch.isnan(loss):
                logger.error("We have NaN numbers, closing\n\n\n")

            loss.backward()
            optimizer.step()

        eval_start = time.time()
        with torch.no_grad():
            filename = '{}_train_{}.pth'.format(dataset_name, time.time())
            nmi, recalls = evaluator.evaluate(model, dl_ev, dataroot=dataset_name, num_classes=train_classes)
            scores.append((epoch, nmi, recalls))

            if recalls[0] > best_recall_at_1:
                best_recall_at_1 = recalls[0]
                best_filename = filename
                torch.save(model.state_dict(), osp.join('./results_nets', filename))
        logger.info('Evaluation took {:.2f}'.format(time.time() - eval_start))
        logger.info('Epoch took {:.2f}s'.format(time.time() - start))
        start = time.time()

    # Evaluation
    with torch.no_grad():
        evaluator.logger.info('TRAINING SCORES (EPOCH, NMI, RECALLS):')
        for epoch, nmi, recalls in scores:
            evaluator.logger.info('{}: {:.3f}, {}'.format(epoch, 100 * nmi, ['{:.3f}'.format(100 * r) for r in recalls]))

        if best_filename != '':
            evaluator.logger.info('Using {}'.format(best_filename))
            model.load_state_dict(torch.load(osp.join('./results_nets', best_filename)))
        
        evaluator.logger.info('FINAL TEST SCORES')
        evaluator.evaluate(model, dl_ev, dataroot=dataset_name, num_classes=train_classes)
        
        filename = '{}_test_{}.pth'.format(dataset_name, time.time())
        torch.save(model.state_dict(), osp.join('./results_nets', filename))
        evaluator.logger.info('Saved final model "{}"'.format(filename))


def get_dataloaders(root, train_classes, labeled_fraction, batch_size, num_workers):
    transform_tr = GL_orig_RE(is_train=True, RE=True)
    data_tr = SubsetDataset(root, range(0, train_classes), labeled_fraction, transform_tr)
    print('Train dataset contains', len(data_tr), 'samples')

    dl_tr = DataLoader(
        data_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    transform_ev = GL_orig_RE(is_train=False, RE=True)
    data_ev = SubsetDataset(root, range(train_classes, 2 * train_classes + 1), 1.0, transform_ev)
    print('Evaluation dataset contains', len(data_ev), 'samples')

    dl_ev = DataLoader(
        data_ev,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
                       
    return dl_tr, dl_ev


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DML Baseline')
    parser.add_argument('--config_path', type=str, default='config/cars.yaml', help='Path to config file')
    args = parser.parse_args()
    
    main(args.config_path)
