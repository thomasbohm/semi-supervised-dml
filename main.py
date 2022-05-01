import torch
import torch.nn as nn
import logging
import os.path as osp
import os
import time
import warnings

from torch.utils.data import DataLoader

from dataset.SubsetDataset import SubsetDataset
from dataset.utils import GL_orig_RE
from evaluation.utils import Evaluator_DML
from net.load_net import load_net
from RAdam import RAdam

warnings.filterwarnings("ignore")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

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

    fh = logging.FileHandler(f'./results/cars_main_{time.time():.2f}.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ### PARAMS ###
    # Dataset
    dataset_name = 'CARS'
    root = f'/gdrive/MyDrive/DML/{dataset_name}'
    #root = f'./data/{dataset_name}'
    train_classes = 98
    labeled_fraction = 0.5
    
    # DataLoader
    batch_size = 32
    num_workers = 4

    # Optimizer
    lr = 0.0001
    weight_decay = 0.000006

    # Training
    epochs = 20
    ##############

    dl_tr, dl_ev = get_dataloaders(
        root,
        train_classes,
        labeled_fraction,
        batch_size,
        num_workers
    )
    
    model, embed_size = load_net(num_classes=train_classes, pretrained_path='no', red=16) # 4=512, 8=256
    model = model.to(device)
    print('Embedding size:', embed_size)

    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    evaluator = Evaluator_DML(device=device)

    start = time.time()
    scores = []
    best_recall_at_1 = 0
    best_filename = ''

    logger.info(f'TRAINING WITH {labeled_fraction * 100}% OF DATA')
    for epoch in range(1, epochs + 1):
        logger.info(f'EPOCH {epoch}/{epochs}')

        model.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            
            preds, embeddings = model(x, output_option='plain')
            loss = loss_fn(preds, y)
            
            if torch.isnan(loss):
                logger.error("We have NaN numbers, closing\n\n\n")
                return 0.0, model

            loss.backward()
            optimizer.step()
                    
        with torch.no_grad():
            filename = f'{dataset_name}_train_{time.time()}.pth'
            nmi, recalls = evaluator.evaluate(model, dl_ev, dataroot=dataset_name, num_classes=train_classes)
            scores.append((epoch, nmi, recalls))

            if recalls[0] > best_recall_at_1:
                best_recall_at_1 = recalls[0]
                best_filename = filename
                torch.save(model.state_dict(), osp.join('./results_nets', filename))

        logger.info(f'epoch took {time.time() - start:.2f}s')
        start = time.time()

    # Evaluation
    with torch.no_grad():
        logger.info('FINAL EVALUATION')
        if best_filename != '':
            model.load_state_dict(torch.load(osp.join('./results_nets', best_filename)))
        
        evaluator.evaluate(model, dl_ev, dataroot=dataset_name, num_classes=train_classes)
        
        filename = f'{dataset_name}_test_{time.time()}.pth'
        torch.save(model.state_dict(), osp.join('./results_nets', filename))

    logger.info('ALL TRAINING SCORES (EPOCH, NMI, RECALLS):')
    for epoch, nmi, recalls in scores:
        logger.info(f'{epoch}, {100 * nmi:.3f}, {[f"{100 * r:.3f}" for r in recalls]}')


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
    main()
