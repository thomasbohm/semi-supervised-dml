import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import random
import json
from torch.utils.data import DataLoader, RandomSampler
from dataset.combine_sampler import CombineSampler
from dataset.m_per_class_sampler import MPerClassSampler
from datetime import datetime
from dataset.ssl_dataset import create_datasets, get_transforms

from net.load_net import load_resnet50
from evaluation.utils import Evaluator
from RAdam import RAdam
from dataset.utils import GL_orig_RE, get_list_of_inds
from dataset.subset_dataset import SubsetDataset


class Trainer():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logging_level = logging.INFO if config['mode'] != 'hyper' else logging.ERROR
        self.evaluator = Evaluator(self.device, logging_level=logging_level)
        self.logger = self.get_logger()
        
        date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.filename = '{}_{}_{}.pth'.format(self.config['dataset']['name'],
                                                 self.config['dataset']['labeled_fraction'] * 100,
                                                 self.config['mode'])
        self.results_dir = './results/{}/{}'.format(config['dataset']['name'], date)
        if not osp.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        self.logger.info('Results saved to "{}"'.format(self.results_dir))
    

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

            model, embed_size = load_resnet50(
                num_classes=self.config['dataset']['train_classes'],
                pretrained_path=self.config['model']['pretrained_path'],
                reduction=self.config['model']['reduction']
            )
            if torch.cuda.device_count() > 1:
                self.logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model = model.to(self.device)
            self.logger.info('Loaded resnet50 with embedding dim {}.'.format(embed_size))

            optimizer = RAdam(model.parameters(),
                              lr=self.config['training']['lr'],
                              weight_decay=self.config['training']['weight_decay'])
            
            loss_fn_lb = nn.CrossEntropyLoss(reduction='none')
            loss_fn_ulb = None
            if 'l2' in self.config['training']['loss'].split('_'):
                loss_fn_ulb = nn.MSELoss(reduction='mean')
            elif 'kl' in self.config['training']['loss'].split('_'):
                loss_fn_ulb = nn.KLDivLoss()

            dl_tr_lb, dl_tr_ulb, dl_ev = self.get_dataloaders_ssl(
                self.config['dataset']['path'],
                self.config['dataset']['train_classes'],
                self.config['dataset']['labeled_fraction'],
                num_workers=4
            )

            if self.config['mode'] != 'test':
                recall_at_1 = self.train_run(
                    model,
                    optimizer,
                    loss_fn_lb,
                    loss_fn_ulb,
                    dl_tr_lb,
                    dl_tr_ulb,
                    dl_ev
                )
                if recall_at_1 > best_recall_at_1:
                    best_recall_at_1 = recall_at_1
                    best_hypers = self.config

                    filename = '{}_{}_best.pth'.format(self.config['dataset']['name'],
                                                       self.config['dataset']['labeled_fraction'] * 100)
                    os.rename(osp.join(self.results_dir, self.filename),
                              osp.join(self.results_dir, filename))
            else:
                self.test_run(model, dl_ev)
        
        if hyper_search:
            self.logger.info('Best R@1: {:.3}'.format(best_recall_at_1 * 100))
            self.logger.info('Best Hyperparameters:\n{}'.format(json.dumps(best_hypers, indent=4)))


    def train_run(self, model, optimizer, loss_fn_lb, loss_fn_ulb, dl_tr_lb, dl_tr_ulb, dl_ev):
        scores = []
        best_epoch = -1
        best_recall_at_1 = 0

        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.logger.info('EPOCH {}/{}'.format(epoch, self.config['training']['epochs']))
            start = time.time()

            if epoch == 30 or epoch == 50:
                self.reduce_lr(model, optimizer)

            for (x_lb, y_lb), (x1_ulb, x2_ulb, _) in zip(dl_tr_lb, dl_tr_ulb):
                optimizer.zero_grad()

                if loss_fn_ulb:
                    x = torch.cat((x_lb, x1_ulb, x2_ulb))
                else:
                    x = x_lb

                x = x.to(self.device)
                preds, embeddings = model(x, output_option='plain')

                preds_lb = preds[:x_lb.shape[0]]
                #preds_lb = F.normalize(preds_lb)
                loss_lb = loss_fn_lb(
                    preds_lb / self.config['training']['temperature'],
                    y_lb.to(self.device)
                )
                self.logger.info('Loss: {}\n{}'.format(loss_lb.shape, loss_lb))
                loss_lb = F.normalize(loss_lb, p=2, dim=-1)
                loss_lb = loss_lb.mean()

                if loss_fn_ulb:
                    preds1_ulb = preds[x_lb.shape[0]:x_lb.shape[0] + x1_ulb.shape[0]]
                    preds2_ulb = preds[x_lb.shape[0] + x1_ulb.shape[0]:]
                    embeddings1_ulb = embeddings[x_lb.shape[0]:x_lb.shape[0] + x1_ulb.shape[0]]
                    embeddings2_ulb = embeddings[x_lb.shape[0] + x1_ulb.shape[0]:]

                    loss_ulb = loss_fn_ulb(embeddings1_ulb, embeddings2_ulb)
                    #loss_ulb = F.normalize(loss_ulb)
                    #loss_ulb = loss_lb.mean()
                    # loss_ulb *= epoch / self.config['training']['epochs']
                else:
                    loss_ulb = torch.tensor(0)
                
                if torch.isnan(loss_lb) or torch.isnan(loss_ulb):
                    self.logger.error("We have NaN numbers, closing\n\n\n")
                    return 0.0

                self.logger.info('loss_lb: {}, loss_ulb: {}'.format(loss_lb, loss_ulb))
                loss = loss_lb + loss_ulb
                loss.backward()
                optimizer.step()

            eval_start = time.time()
            with torch.no_grad():
                recalls, nmi = self.evaluator.evaluate(
                    model,
                    dl_ev,
                    dataroot=self.config['dataset']['name'],
                    num_classes=self.config['dataset']['train_classes']
                )
                scores.append((epoch, recalls, nmi))

                if recalls[0] > best_recall_at_1:
                    best_recall_at_1 = recalls[0]
                    best_epoch = epoch
                    torch.save(model.state_dict(), osp.join(self.results_dir, self.filename))

            self.evaluator.logger.info('Eval  took {:.0f}s'.format(time.time() - eval_start))
            self.logger.info('Epoch took {:.0f}s'.format(time.time() - start))
            start = time.time()

        self.logger.info('-' * 50)
        self.logger.info('ALL TRAINING SCORES (EPOCH, RECALLS, NMI):')
        for epoch, recalls, nmi in scores:
            self.logger.info('{}: {}, {:.1f}'.format(
                epoch,
                ['{:.1f}'.format(100 * r) for r in recalls],
                100 * nmi)
            )
        self.logger.info('BEST R@1 (EPOCH {}): {:.3f}'.format(best_epoch, best_recall_at_1))

        return best_recall_at_1

    
    def test_run(self, model, dl_ev):
        with torch.no_grad():
            recalls, nmi = self.evaluator.evaluate(
                model,
                dl_ev,
                dataroot=self.config['dataset']['name'],
                num_classes=self.config['dataset']['train_classes']
            )
            return recalls, nmi


    def get_logger(self):
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(name)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger
    

    def get_dataloaders_ssl(self, data_path, train_classes, labeled_fraction, num_workers):
        trans_train, trans_train_strong, trans_eval = get_transforms()
        self.logger.info('Train transform:\n{}'.format(trans_train))
        self.logger.info('Train transform strong:\n{}'.format(trans_train_strong))
        self.logger.info('Eval transform:\n{}'.format(trans_eval))

        dset_lb, dset_ulb, dset_eval = create_datasets(
            data_path,
            train_classes,
            labeled_fraction,
            trans_train,
            trans_train_strong,
            trans_eval
        )
        self.logger.info('{} train samples ({} labeled + {} unlabeled)'.format(
            len(dset_lb) + len(dset_ulb),
            len(dset_lb),
            len(dset_ulb))
        )
        self.logger.info('{} eval samples'.format(len(dset_eval)))

        class_per_batch = self.config['training']['num_classes_iter']
        elements_per_class = self.config['training']['num_elements_class']

        batch_size_lb = class_per_batch * elements_per_class
        batch_size_ulb = self.config['training']['ulb_batch_size_factor'] * batch_size_lb
        num_batches = len(dset_ulb) // batch_size_ulb

        dl_train_lb, dl_train_ulb = None, None

        if self.config['mode'] != 'test':
            sampler_lb = MPerClassSampler(
                dset_lb.targets,
                m=elements_per_class,
                batch_size=batch_size_lb,
                length_before_new_iter=batch_size_lb * num_batches
            )
            dl_train_lb = DataLoader(
                dset_lb,
                batch_size=batch_size_lb,
                sampler=sampler_lb,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True
            )
            sampler_ulb = RandomSampler(
                dset_ulb,
                replacement=True,
                num_samples=batch_size_ulb * num_batches
            )
            dl_train_ulb = DataLoader(
                dset_ulb,
                batch_size=batch_size_ulb,
                sampler=sampler_ulb,
                num_workers=num_workers,
                drop_last=True,
                pin_memory=True
            )
            self.logger.info('Batch size labeled:   {}'.format(batch_size_lb))
            self.logger.info('Batch size unlabeled: {}'.format(batch_size_ulb))
            self.logger.info('Num batches: {}'.format(num_batches))
            assert len(dl_train_lb) == len(dl_train_ulb) == num_batches

        dl_eval = DataLoader(
            dset_eval,
            batch_size=batch_size_lb,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        return dl_train_lb, dl_train_ulb, dl_eval
        

    def sample_hypers(self):
        config = {
            'lr': 10 ** random.uniform(-5, -3),
            'weight_decay': 10 ** random.uniform(-15, -6),
            'num_classes_iter': random.randint(6, 15),
            'num_elements_class': random.randint(3, 9),
            'temperature': random.random(),
            'epochs': 40
        }
        self.config['training'].update(config)

    
    def reduce_lr(self, model, optimizer):
        self.logger.info("Reducing learning rate:")
        model.load_state_dict(torch.load(osp.join(self.results_dir, self.filename)))
        for g in optimizer.param_groups:
            self.logger.info('{} -> {}'.format(g['lr'], g['lr'] / 10))
            g['lr'] /= 10.
