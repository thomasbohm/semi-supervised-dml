import logging
import time
import torch
import torch.nn as nn
import os
import os.path as osp
import random
import json
from torch.utils.data import DataLoader
from dataset.combine_sampler import CombineSampler

from net.load_net import load_resnet50
from evaluation.utils import Evaluator
from RAdam import RAdam
from dataset.utils import GL_orig_RE, get_list_of_inds
from dataset.subset_dataset import SubsetDataset


class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.evaluator = Evaluator(self.device)
        self.logger = self.get_logger()
        self.filename = '{}_{}_train_{:.0f}.pth'.format(self.config['dataset']['name'],
                                                        self.config['dataset']['labeled_fraction'] * 100,
                                                        time.time())

        self.results_nets_dir = './results_nets_train/{}'.format(config['dataset']['name'])
        if not osp.isdir(self.results_nets_dir):
            os.makedirs(self.results_nets_dir)

        self.results_nets_dir_final = './results_nets/{}'.format(config['dataset']['name'])
        if not osp.isdir(self.results_nets_dir_final):
            os.makedirs(self.results_nets_dir_final)
    

    def start(self):
        self.logger.info('Device: {}'.format(self.device))
        hyper_search = self.config['mode'] == 'hyper'
        num_runs = 30 if hyper_search else 1

        best_recall_at_1 = -1
        best_hypers = self.config
        for run in range(1, num_runs + 1):
            if hyper_search:
                self.logger.info('Search run: {}/{}'.format(run, num_runs))
                self.sample_hypers()

            self.logger.info("Config:\n{}".format(json.dumps(self.config, indent=4)))

            model, embed_size = load_resnet50(num_classes=self.config['dataset']['train_classes'],
                                              pretrained_path=self.config['model']['pretrained_path'],
                                              reduction=self.config['model']['reduction'])
            model = model.to(self.device)
            self.logger.info('Loaded model with embedding dim {}.'.format(embed_size))

            optimizer = RAdam(model.parameters(),
                              lr=self.config['training']['lr'],
                              weight_decay=self.config['training']['weight_decay'])
            loss_fn = nn.CrossEntropyLoss()

            dl_tr, dl_ev = self.get_dataloaders(
                self.config['dataset']['path'],
                self.config['dataset']['name'],
                self.config['dataset']['train_classes'],
                self.config['dataset']['labeled_fraction'],
                num_workers=4
            )

            if self.config['mode'] != 'test':
                recall_at_1 = self.train(model, optimizer, loss_fn, dl_tr, dl_ev)
                if recall_at_1 > best_recall_at_1:
                    best_recall_at_1 = recall_at_1
                    best_hypers = self.config

                    filename = '{}_{}_{:.3f}.pth'.format(self.config['dataset']['name'],
                                                        self.config['dataset']['labeled_fraction'] * 100,
                                                        best_recall_at_1 * 100)
                    os.rename(osp.join(self.results_nets_dir, self.filename),
                            osp.join(self.results_nets_dir_final, filename))
            else:
                self.evaluate(model, dl_ev)
        
        if hyper_search:
            self.logger.info('Best R@1: {:.3}'.format(best_recall_at_1 * 100))
            self.logger.info('Best Hyperparameters:\n{}'.format(json.dumps(best_hypers, indent=4, sort_keys=True)))
        self.logger.info('-' * 50)


    def train(self, model, optimizer, loss_fn, dl_tr, dl_ev):
        scores = []
        best_epoch = -1
        best_recall_at_1 = 0

        self.logger.info('TRAINING WITH {}% OF DATA'.format(self.config['dataset']['labeled_fraction'] * 100))
        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.logger.info('EPOCH {}/{}'.format(epoch, self.config['training']['epochs']))
            start = time.time()

            if epoch == 31 or epoch == 51:
                self.reduce_lr(model, optimizer)

            model.train()
            for x, y in dl_tr:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                
                preds, embeddings = model(x, output_option='plain')
                loss = loss_fn(preds / self.config['training']['temperatur'], y)
                
                if torch.isnan(loss):
                    self.logger.error("We have NaN numbers, closing\n\n\n")

                loss.backward()
                optimizer.step()

            eval_start = time.time()
            with torch.no_grad():
                nmi, recalls = self.evaluator.evaluate(model,
                                                       dl_ev,
                                                       dataroot=self.config['dataset']['name'],
                                                       num_classes=self.config['dataset']['train_classes'])
                scores.append((epoch, nmi, recalls))

                if recalls[0] > best_recall_at_1:
                    best_recall_at_1 = recalls[0]
                    best_epoch = epoch
                    torch.save(model.state_dict(), osp.join(self.results_nets_dir, self.filename))

            self.logger.info('Evaluation took {:.2f}'.format(time.time() - eval_start))
            self.logger.info('Epoch took {:.2f}s'.format(time.time() - start))
            start = time.time()

        self.evaluator.logger.info('ALL TRAINING SCORES (EPOCH, NMI, RECALLS):')
        for epoch, nmi, recalls in scores:
            self.evaluator.logger.info('{}: {:.3f}, {}'.format(epoch, 100 * nmi, ['{:.3f}'.format(100 * r) for r in recalls]))
        self.evaluator.logger.info('BEST R@1 IN EPOCH {}'.format(best_epoch))

        return best_recall_at_1

    
    def evaluate(self, model, dl_ev):
        with torch.no_grad():
            nmi, recalls = self.evaluator.evaluate(model,
                                                   dl_ev,
                                                   dataroot=self.config['dataset']['name'],
                                                   num_classes=self.config['dataset']['train_classes'])
            return nmi, recalls

    def get_logger(self):
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(name)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
    

    def get_dataloaders(self, root, dataset_name, train_classes, labeled_fraction, num_workers):
        batch_size = self.config['training']['num_classes_iter'] * self.config['training']['num_elements_class']
        self.logger.info('Batch size: {}'.format(batch_size))
        
        transform_tr = GL_orig_RE(is_train=True, RE=self.config['dataset']['random_erasing'])
        data_tr = SubsetDataset(root, range(0, train_classes), labeled_fraction, transform_tr)
        self.logger.info('Train dataset contains {} samples.'.format(len(data_tr)))

        sampler = CombineSampler(get_list_of_inds(data_tr),
                                 self.config['training']['num_classes_iter'],
                                 self.config['training']['num_elements_class'])

        dl_tr = DataLoader(
            data_tr,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True
        )
        
        all_classes = 2 * train_classes
        if dataset_name == 'SOP':
            all_classes -= 2

        transform_ev = GL_orig_RE(is_train=False, RE=self.config['dataset']['random_erasing'])
        data_ev = SubsetDataset(root, range(train_classes, all_classes + 1), 1.0, transform_ev)
        self.logger.info('Evaluation dataset contains {} samples.'.format(len(data_ev)))

        dl_ev = DataLoader(
            data_ev,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
                        
        return dl_tr, dl_ev
    

    def sample_hypers(self):
        config = {
            'lr': 10 ** random.uniform(-5, -3),
            'weight_decay': 10 ** random.uniform(-15, -6),
            'num_classes_iter': random.randint(6, 15),
            'num_elements_class': random.randint(3, 9),
            'temperatur': random.random(),
            'epochs': 40
        }
        self.config['training'].update(config)

    
    def reduce_lr(self, model, optimizer):
        self.logger.info("Reducing learning rate:")
        model.load_state_dict(torch.load(osp.join(self.results_nets_dir, self.filename)))
        for g in optimizer.param_groups:
            self.logger.info('{} -> {}'.format(g['lr'], g['lr'] / 10))
            g['lr'] /= 10.
