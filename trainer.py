import copy
import json
import logging
import os
import os.path as osp
import random
import time
from datetime import datetime
from typing import Optional, Tuple, Union
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, RandomSampler

from dataset.m_per_class_sampler import MPerClassSampler
from dataset.ssl_dataset import create_datasets, get_transforms
from evaluation.utils import Evaluator
from net.gnn import GNNModel
from net.load_net import load_resnet50
from RAdam import RAdam
from net.loss import NTXentLoss


class Trainer():
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if config['mode'] != 'hyper':
            logging_level = logging.INFO
        else:
            logging_level = logging.ERROR
        self.evaluator = Evaluator(self.device, logging_level=logging_level)
        self.logger = self.get_logger()

        date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.filename = '{}_{}_{}.pth'.format(
            self.config['dataset']['name'],
            self.config['dataset']['labeled_fraction'] * 100,
            self.config['mode']
        )
        self.filename_gnn = '{}_{}_{}_gnn.pth'.format(
            self.config['dataset']['name'],
            self.config['dataset']['labeled_fraction'] * 100,
            self.config['mode']
        )
        self.results_dir = f'./results/{config["dataset"]["name"]}/{date}/'
        if not osp.isdir(self.results_dir):
            os.makedirs(self.results_dir)
        self.logger.info(f'Results saved to "{self.results_dir}"')

        self.labeled_only = self.config['training']['loss_ulb'] == '' or \
            self.config['dataset']['labeled_fraction'] >= 1.0

    def start(self):
        self.logger.info(f'Device: {self.device}')
        hyper_search = self.config['mode'] == 'hyper'
        num_runs = 30 if hyper_search else 1

        best_run = -1
        best_recall_at_1 = -1
        best_hypers = {}
        for run in range(1, num_runs + 1):
            if hyper_search:
                self.logger.info(f'Current best: Run {best_run} | R@1: {best_recall_at_1:.2f}')
                self.logger.info(f'Search run: {run}/{num_runs}')
                self.sample_hypers()

            self.logger.info(f'Config:\n{json.dumps(self.config, indent=4)}')

            seed_everything()

            # ResNet
            num_classes = self.config['dataset']['train_classes']
            resnet, embed_dim = load_resnet50(
                num_classes=num_classes,
                pretrained_path=self.config['resnet']['pretrained_path'],
                reduction=self.config['resnet']['reduction'],
                neck=self.config['resnet']['bottleneck']
            )
            if torch.cuda.device_count() > 1:
                self.logger.info(f'Using {torch.cuda.device_count()} GPUs')
                resnet = nn.parallel.DataParallel(resnet)
            resnet = resnet.to(self.device)
            self.logger.info(f'Loaded resnet50 with embedding dim {embed_dim}.')

            # Projection Head
            head_ulb = None
            if self.config['training']['loss_ulb'] in ['l2_head', 'huber_head', 'kl_head', 'simclr']:
                if self.config['training']['loss_ulb'] in ['l2_head', 'huber_head', 'simclr']:
                    head_ulb = nn.Sequential(
                        nn.Linear(embed_dim, 2 * embed_dim),
                        nn.ReLU(),
                        nn.Linear(2 * embed_dim, embed_dim)
                    )
                else:
                    head_ulb = nn.Sequential(
                        nn.Linear(num_classes, 4 * num_classes),
                        nn.ReLU(),
                        nn.Linear(4 * num_classes, num_classes)
                    )
                if torch.cuda.device_count() > 1:
                    head_ulb = nn.parallel.DataParallel(head_ulb)
                head_ulb = head_ulb.to(self.device)
                params = list(set(resnet.parameters())) + list(set(head_ulb.parameters()))
            else:
                params = resnet.parameters()

            gnn = None
            loss_fn_gnn = None
            if 'gnn' in self.config['model'].split('_'):
                gnn = GNNModel(
                    self.device,
                    embed_dim = embed_dim,
                    output_dim = num_classes,
                    num_layers = self.config['gnn']['num_layers'],
                    num_heads = self.config['gnn']['num_heads'],
                    num_proxies = self.config['gnn']['num_proxies'],
                    add_mlp = self.config['gnn']['add_mlp'],
                    gnn_conv = self.config['gnn']['gnn_conv'],
                    gnn_fc = self.config['gnn']['gnn_fc'],
                    reduction_layer = self.config['gnn']['reduction_layer']
                ).to(self.device)

                if self.config['gnn']['pretrained_path'] not in ['', 'no']:
                    gnn.load_state_dict(torch.load(self.config['gnn']['pretrained_path']))

                self.logger.info(gnn)
                params = [
                    { 'params': resnet.parameters() },
                    { 'params': gnn.parameters(), 'lr': self.config['training']['lr_gnn'] }
                ]
                loss_fn_gnn = nn.CrossEntropyLoss(reduction='none')

            # Optimizer
            optimizer = RAdam(
                params,
                lr=self.config['training']['lr'],
                weight_decay=self.config['training']['weight_decay']
            )

            # Labeled Loss Function
            if self.config['training']['loss_lb'] == 'lsce':
                loss_fn_lb = nn.CrossEntropyLoss(label_smoothing=0.1)
            else:
                loss_fn_lb = nn.CrossEntropyLoss()

            # Unlabeled Loss Function
            if self.config['training']['loss_ulb'] == '':
                loss_fn_ulb = None
            elif self.config['training']['loss_ulb'] in ['l2', 'l2_head']:
                loss_fn_ulb = nn.MSELoss()
            elif self.config['training']['loss_ulb'] in ['kl', 'kl_head']:
                loss_fn_ulb = nn.KLDivLoss(log_target=True, reduction='batchmean')
            elif self.config['training']['loss_ulb'] in ['huber', 'huber_head']:
                loss_fn_ulb = nn.HuberLoss()
            elif self.config['training']['loss_ulb'] in ['ce_soft', 'ce_hard']:
                loss_fn_ulb = nn.CrossEntropyLoss()
            elif self.config['training']['loss_ulb'] in ['ce_thresh']:
                loss_fn_ulb = nn.CrossEntropyLoss(reduction='none')
            elif self.config['training']['loss_ulb'] == 'simclr':
                class_per_batch = self.config['training']['num_classes_iter']
                elements_per_class = self.config['training']['num_elements_class']
                batch_size_lb = class_per_batch * elements_per_class
                batch_size_ulb = self.config['training']['ulb_batch_size_factor'] * batch_size_lb
                loss_fn_ulb = NTXentLoss(batch_size=batch_size_ulb, device=self.device)
            else:
                self.logger.error(f'Unlabeled loss not supported: {self.config["training"]["loss_ulb"]}')
                return

            dl_tr_lb, dl_tr_ulb, dl_ev = self.get_dataloaders_ssl(
                self.config['dataset']['path'],
                self.config['dataset']['train_classes'],
                self.config['dataset']['labeled_fraction'],
                num_workers=self.config['dataset']['num_workers']
            )

            if self.config['mode'] != 'test':
                assert dl_tr_lb
                recall_at_1 = self.train_run(
                    model=resnet,
                    head_ulb=head_ulb,
                    optimizer=optimizer,
                    loss_fn_lb=loss_fn_lb,
                    loss_fn_ulb=loss_fn_ulb,
                    dl_tr_lb=dl_tr_lb,
                    dl_tr_ulb=dl_tr_ulb,
                    dl_ev=dl_ev,
                    gnn_model=gnn,
                    gnn_loss_fn=loss_fn_gnn
                )
                if recall_at_1 > best_recall_at_1:
                    best_run = run
                    best_recall_at_1 = recall_at_1
                    best_hypers = copy.deepcopy(self.config)

                    filename = '{}_{}_best.pth'.format(
                        self.config['dataset']['name'],
                        self.config['dataset']['labeled_fraction'] * 100
                    )
                    filename_gnn = '{}_{}_gnn_best.pth'.format(
                        self.config['dataset']['name'],
                        self.config['dataset']['labeled_fraction'] * 100
                    )
                    os.rename(osp.join(self.results_dir, self.filename),
                              osp.join(self.results_dir, filename))
                    if gnn:
                        os.rename(
                            osp.join(self.results_dir, self.filename_gnn),
                            osp.join(self.results_dir, filename_gnn)
                        )
                    if gnn and self.config['mode'] == 'train':
                        resnet.load_state_dict(torch.load(osp.join(self.results_dir, filename)))
                        gnn.load_state_dict(torch.load(osp.join(self.results_dir, filename_gnn)))

            if self.config['mode'] != 'hyper':
                plots_dir = osp.join(self.results_dir, 'plots')
                os.mkdir(plots_dir)
                self.test_run(resnet, dl_ev, plots_dir, gnn)
                if gnn and dl_tr_lb and dl_tr_ulb:
                    self.evaluator.create_train_plots(
                        resnet,
                        gnn,
                        dl_tr_lb,
                        dl_tr_ulb,
                        self.config['dataset']['train_classes'],
                        plots_dir
                    )

        if hyper_search:
            self.logger.info(f'Best Run: {best_run}')
            self.logger.info(f'Best R@1: {best_recall_at_1 * 100:.4}')
            self.logger.info(f'Best Hyperparameters:\n{json.dumps(best_hypers, indent=4)}')

    def train_run(
        self,
        model: nn.Module,
        head_ulb: Optional[nn.Module],
        optimizer: Optimizer,
        loss_fn_lb: nn.CrossEntropyLoss,
        loss_fn_ulb: Optional[Union[nn.MSELoss, nn.KLDivLoss, nn.HuberLoss, NTXentLoss, nn.CrossEntropyLoss]],
        dl_tr_lb: DataLoader,
        dl_tr_ulb: Optional[DataLoader],
        dl_ev: DataLoader,
        gnn_model: Optional[GNNModel],
        gnn_loss_fn: Optional[nn.CrossEntropyLoss]
    ):
        scores = []
        best_epoch = -1
        best_recall_at_1 = 0.0

        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.logger.info(f'EPOCH {epoch}/{self.config["training"]["epochs"]}')
            start = time.time()

            if epoch == 30 or epoch == 50:
                self.reduce_lr(model, optimizer)

            if self.labeled_only:
                self.train_epoch_without_ulb(
                    dl_tr_lb,
                    model,
                    optimizer,
                    loss_fn_lb,
                    epoch,
                    self.config['mode'] == 'train',
                    gnn_model,
                    gnn_loss_fn
                )
            else:
                assert dl_tr_ulb and loss_fn_ulb
                self.train_epoch_with_ulb(
                    dl_tr_lb,
                    dl_tr_ulb,
                    model,
                    head_ulb,
                    optimizer,
                    loss_fn_lb,
                    loss_fn_ulb,
                    epoch,
                    self.config['mode'] == 'train',
                    gnn_model,
                    gnn_loss_fn
                )
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
                    torch.save(
                        model.state_dict(),
                        osp.join(self.results_dir, self.filename)
                    )
                    if 'gnn' in self.config['model'].split('_') and gnn_model:
                        torch.save(
                            gnn_model.state_dict(),
                            osp.join(self.results_dir, self.filename_gnn)
                        )

            self.evaluator.logger.info(f'Eval took {time.time() - eval_start:.0f}s')
            self.logger.info(f'Epoch took {time.time() - start:.0f}s')

        self.logger.info('-' * 50)
        self.logger.info('ALL TRAINING SCORES (EPOCH, RECALLS, NMI):')
        for epoch, recalls, nmi in scores:
            self.logger.info('{}: {}, {:.1f}'.format(
                epoch,
                ['{:.1f}'.format(100 * r) for r in recalls],
                100 * nmi
            ))
        self.logger.info(f'BEST R@1 (EPOCH {best_epoch}): {best_recall_at_1:.3f}')

        return best_recall_at_1

    def train_epoch_without_ulb(
        self,
        dl_tr_lb: DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn_lb: nn.Module,
        epoch: int,
        plot_tsne: bool,
        gnn_model: Optional[GNNModel],
        gnn_loss_fn: Optional[nn.CrossEntropyLoss]
    ):
        temp = self.config['training']['loss_lb_temp']
        first_batch = True
        for (x, y, p) in dl_tr_lb:
            optimizer.zero_grad()

            x = x.to(self.device)
            y = y.to(self.device)
            preds, embeddings = model(x, output_option='norm', val=False)
            loss = loss_fn_lb(preds / temp, y)

            if 'gnn' in self.config['model'].split('_'):
                assert gnn_model and gnn_loss_fn
                torch.use_deterministic_algorithms(False)
                preds, embeddings = gnn_model(embeddings)
                torch.use_deterministic_algorithms(True)
                loss_gnn = gnn_loss_fn(preds, y).mean()
                loss += loss_gnn

            #if plot_tsne and epoch % 10 == 0 and first_batch:
                #path = osp.join(self.results_dir, f'tsne_train_lb_{epoch}.svg')
                #self.evaluator.create_tsne_plot(embeddings, y, path)
                # self.logger.info(f'Counter(y_lb): {Counter(y.tolist()).most_common()}')

            if torch.isnan(loss):
                self.logger.error('We have NaN numbers, closing\n\n\n')
                return

            # self.logger.info('loss_lb: {}'.format(loss))
            loss.backward()
            optimizer.step()
            first_batch = False




    def train_epoch_with_ulb(
        self,
        dl_tr_lb: DataLoader,
        dl_tr_ulb: DataLoader,
        encoder: nn.Module,
        head_ulb: Optional[nn.Module],
        optimizer: Optimizer,
        loss_fn_lb: nn.Module,
        loss_fn_ulb: nn.Module,
        epoch: int,
        plot_tsne: bool,
        gnn_model: Optional[GNNModel],
        gnn_loss_fn: Optional[nn.CrossEntropyLoss]
    ):
        temp = self.config['training']['loss_lb_temp']
        first_batch = True
        for (x_lb, y_lb, p_lb), (x_ulb_w, x_ulb_s, y_ulb, p_ulb) in zip(dl_tr_lb, dl_tr_ulb):
            optimizer.zero_grad()

            x = torch.cat((x_lb, x_ulb_w, x_ulb_s)).to(self.device)
            preds, embeddings = encoder(x, output_option='norm', val=False)

            # Labeled encoder loss
            preds_lb = preds[:x_lb.shape[0]]
            y_lb = y_lb.to(self.device)
            loss_lb = loss_fn_lb(preds_lb / temp, y_lb)

            # Unlabeled encoder loss
            preds_ulb_w = preds[x_lb.shape[0] : x_lb.shape[0] + x_ulb_w.shape[0]]
            preds_ulb_s = preds[x_lb.shape[0] + x_ulb_w.shape[0] :]
            preds_ulb_w = F.softmax(preds_ulb_w)
            preds_ulb_w_max, y_ulb_w = preds_ulb_w.max(dim=1)
            mask = preds_ulb_w_max.gt(self.config['training']['loss_ulb_threshold'])
            loss_ulb = loss_fn_ulb(preds_ulb_s, y_ulb_w) * mask
            loss_ulb = loss_ulb.mean()
            
            loss = loss_lb + self.config['training']['loss_ulb_weight'] * loss_ulb
            if torch.isnan(loss):
                self.logger.error('We have NaN numbers, closing\n\n\n')
                return

            if gnn_model and gnn_loss_fn:
                if self.config['gnn']['batch_proxies']:
                    proxy_idx = torch.cat((y_lb, y_ulb_w[preds_ulb_w_max > self.config['training']['loss_ulb_threshold']])).unique()
                else:
                    proxy_idx = None
                torch.use_deterministic_algorithms(False)
                preds_gnn, embeds_gnn, preds_proxies, embeds_proxies = gnn_model(
                    embeddings,
                    #true_proxy=torch.cat((y_lb, y_ulb_w, y_ulb_w)),
                    return_proxies=True,
                    proxy_idx=proxy_idx,
                    #kclosest=self.config['training']['num_classes_iter']
                )
                torch.use_deterministic_algorithms(True)

                preds_gnn_lb = preds_gnn[:x_lb.shape[0]]
                preds_gnn_ulb_w = preds_gnn[x_lb.shape[0] : x_lb.shape[0] + x_ulb_w.shape[0]]
                preds_gnn_ulb_s = preds_gnn[x_lb.shape[0] + x_ulb_w.shape[0] :]

                preds_gnn_ulb_w = F.softmax(preds_gnn_ulb_w)
                preds_gnn_max, preds_gnn_argmax = preds_gnn_ulb_w.max(dim=1)
                mask_gnn = preds_gnn_max.gt(self.config['training']['loss_ulb_gnn_threshold'])
                mask_gnn = torch.cat((torch.ones(x_lb.shape[0], device=self.device), mask_gnn))

                x_gnn = torch.cat((preds_gnn_lb, preds_gnn_ulb_s))
                y_gnn = torch.cat((y_lb, preds_gnn_argmax))

                loss_gnn = gnn_loss_fn(x_gnn, y_gnn) * mask_gnn
                loss_gnn = loss_gnn.mean()
                loss += loss_gnn

                loss_proxies = 0.0
                if self.config['training']['loss_proxy'] == 'l2':
                    embeds_gnn_lb = embeds_gnn[:x_lb.shape[0]]
                    embeds_gnn_ulb_s = embeds_gnn[x_lb.shape[0] + x_ulb_w.shape[0] :]
                    embeds = torch.cat((embeds_gnn_lb, embeds_gnn_ulb_s))
                    proxies = torch.index_select(gnn_model.proxies, 0, y_gnn)
                    loss_proxies = F.mse_loss(embeds, proxies, reduction='none') * mask_gnn.unsqueeze(1)
                    loss_proxies = loss_proxies.mean()
                    loss += self.config['training']['loss_proxy_weight'] * loss_proxies
                elif self.config['training']['loss_proxy'] == 'ce':
                    classes = y_gnn.unique()
                    loss_proxies = F.cross_entropy(preds_proxies[classes], classes)
                    loss += loss_proxies

                if first_batch:
                    self.logger.info(f'ResNet lb : {loss_lb:.2f}')
                    self.logger.info(f'ResNet ulb: {self.config["training"]["loss_ulb_weight"] * loss_ulb:.2f}')
                    self.logger.info(f'GNN       : {loss_gnn:.2f}')
                    self.logger.info(f'GNN proxy : {self.config["training"]["loss_proxy_weight"] * loss_proxies:.2f}')
                    self.logger.info(f'Total loss: {loss:.2f}')

            loss.backward()
            optimizer.step()
            first_batch = False







    '''
    def train_epoch_with_ulb_old(
        self,
        dl_tr_lb: DataLoader,
        dl_tr_ulb: DataLoader,
        model: nn.Module,
        head_ulb: Optional[nn.Module],
        optimizer: Optimizer,
        loss_fn_lb: nn.Module,
        loss_fn_ulb: nn.Module,
        epoch: int,
        plot_tsne: bool,
        gnn_model: Optional[GNNModel],
        gnn_loss_fn: Optional[nn.CrossEntropyLoss]
    ):
        temp = self.config['training']['loss_lb_temp']
        first_batch = True
        for (x_lb, y_lb, p_lb), (x_ulb_w, x_ulb_s, y_ulb, p_ulb) in zip(dl_tr_lb, dl_tr_ulb):
            optimizer.zero_grad()

            x = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            x = x.to(self.device)
            preds, embeddings = model(x, output_option='norm', val=False)

            preds_lb = preds[:x_lb.shape[0]]
            if plot_tsne and epoch % 10 == 0 and first_batch:
                self.evaluator.create_tsne_plot(
                    preds_lb,
                    y_lb,
                    osp.join(self.results_dir, f'tsne_train_lb_{epoch}.png')
                )
                # self.logger.info(f'Counter(y_lb): {Counter(y_lb.tolist()).most_common()}')
            loss_lb = loss_fn_lb(preds_lb / temp, y_lb.to(self.device))

            # embedding based losses
            if self.config['training']['loss_ulb'] in ['l2', 'huber', 'l2_head', 'huber_head', 'simclr']:
                if self.config['training']['loss_ulb'] in ['l2', 'huber']:
                    embeddings_ulb_w = embeddings[x_lb.shape[0]:x_lb.shape[0] + x_ulb_w.shape[0]]
                    embeddings_ulb_s = embeddings[x_lb.shape[0] + x_ulb_w.shape[0]:]
                else:
                    assert head_ulb
                    embeddings_ulb = embeddings[x_lb.shape[0]:]
                    embeddings_ulb = head_ulb(embeddings_ulb)
                    embeddings_ulb_w = embeddings_ulb[:x_ulb_w.shape[0]]
                    embeddings_ulb_s = embeddings_ulb[x_ulb_w.shape[0]:]
                if plot_tsne and epoch % 10 == 0 and first_batch:
                    self.evaluator.create_tsne_plot(
                        embeddings_ulb_w,
                        y_ulb,
                        osp.join(self.results_dir, f'tsne_train_ulb_w_{epoch}.png')
                    )
                    self.evaluator.create_tsne_plot(embeddings_ulb_s,
                        y_ulb,
                        osp.join(self.results_dir, f'tsne_train_ulb_s_{epoch}.png')
                    )
                loss_ulb = loss_fn_ulb(embeddings_ulb_w, embeddings_ulb_s)
            
            # prediction based losses: 'kl', 'kl_head', 'ce_soft', 'ce_hard', 'ce_thresh'
            else:
                if self.config['training']['loss_ulb'] in ['kl', 'ce_soft', 'ce_hard', 'ce_thresh']:
                    preds_ulb_w = preds[x_lb.shape[0] : x_lb.shape[0] + x_ulb_w.shape[0]]
                    preds_ulb_s = preds[x_lb.shape[0] + x_ulb_w.shape[0] :]
                elif self.config['training']['loss_ulb'] == 'kl_head':
                    assert head_ulb
                    preds_ulb = preds[x_lb.shape[0]:]
                    preds_ulb = head_ulb(preds_ulb)
                    preds_ulb_w = preds_ulb[:x_ulb_w.shape[0]]
                    preds_ulb_s = preds_ulb[x_ulb_w.shape[0]:]
                else:
                    self.logger.error(f'Unlabeled loss not supported: {self.config["training"]["loss_ulb"]}')
                    return

                if plot_tsne and epoch % 10 == 0 and first_batch:
                    self.evaluator.create_tsne_plot(
                        preds_ulb_w,
                        y_ulb,
                        osp.join(self.results_dir, f'tsne_train_ulb_w_{epoch}.png')
                    )
                    self.evaluator.create_tsne_plot(preds_ulb_s,
                        y_ulb,
                        osp.join(self.results_dir, f'tsne_train_ulb_s_{epoch}.png')
                    )

                if self.config['training']['loss_ulb'] in ['kl', 'kl_head']:
                    preds_ulb_w = F.log_softmax(preds_ulb_w)
                    preds_ulb_s = F.log_softmax(preds_ulb_s)
                elif self.config['training']['loss_ulb'] in ['ce_soft', 'ce_hard']:
                    if self.config['training']['loss_ulb'] == 'ce_soft':
                        preds_ulb_w = F.softmax(preds_ulb_w)
                        preds_ulb_s = preds_ulb_s / self.config['training']['loss_ulb_temp']
                    else:
                        preds_ulb_w = preds_ulb_w.argmax(dim=1)

                if self.config['training']['loss_ulb'] == 'ce_thresh':
                    preds_ulb_w = F.softmax(preds_ulb_w)
                    preds_max, preds_argmax = preds_ulb_w.max(dim=1)
                    mask = preds_max.gt(self.config['training']['loss_ulb_threshold'])
                    loss_ulb = loss_fn_ulb(preds_ulb_s, preds_argmax) * mask
                    loss_ulb = loss_ulb.mean()
                else:
                    loss_ulb = loss_fn_ulb(preds_ulb_s, preds_ulb_w)

            # warm up
            if epoch < self.config['training']['loss_ulb_warmup']:
                loss_ulb *= epoch / self.config['training']['loss_ulb_warmup']

            if torch.isnan(loss_lb) or torch.isnan(loss_ulb):
                self.logger.error('We have NaN numbers, closing\n\n\n')
                return

            loss_ulb *= self.config['training']['loss_ulb_weight']
            loss = loss_lb + loss_ulb

            if gnn_model and gnn_loss_fn:
                torch.use_deterministic_algorithms(False)
                preds_gnn, embeddings_gnn, preds_proxies, embeds_proxies = gnn_model(embeddings, return_proxies=True)
                torch.use_deterministic_algorithms(True)

                preds_gnn_lb = preds_gnn[:x_lb.shape[0]]
                preds_gnn_ulb_w = preds_gnn[x_lb.shape[0] : x_lb.shape[0] + x_ulb_w.shape[0]]
                preds_gnn_ulb_s = preds_gnn[x_lb.shape[0] + x_ulb_w.shape[0] :]

                preds_gnn_ulb_w = F.softmax(preds_gnn_ulb_w)
                preds_gnn_max, preds_gnn_argmax = preds_gnn_ulb_w.max(dim=1)
                mask_gnn = preds_gnn_max.gt(self.config['training']['loss_ulb_gnn_threshold'])
                #self.logger.info(f'preds_gnn_ulb_w > threshold: {mask_gnn.sum()/preds_gnn_ulb_w.shape[0] * 100:.2f}%')
                mask_gnn = torch.cat((torch.ones(x_lb.shape[0], device=self.device), mask_gnn))

                x_gnn = torch.cat((preds_gnn_lb, preds_gnn_ulb_s))
                y_gnn = torch.cat((y_lb.to(self.device), preds_gnn_argmax))

                loss_gnn = gnn_loss_fn(x_gnn, y_gnn) * mask_gnn
                loss_gnn = loss_gnn.mean()
                loss += loss_gnn

                loss_proxies = 0.0
                if self.config['training']['loss_proxy']:
                    classes = y_gnn.unique()
                    loss_proxies = F.cross_entropy(preds_proxies[classes], classes)
                    loss += loss_proxies
                if first_batch:
                    self.logger.info(f'ResNet lb : {loss_lb:.2f}')
                    self.logger.info(f'ResNet ulb: {loss_ulb:.2f}')
                    self.logger.info(f'GNN       : {loss_gnn:.2f}')
                    self.logger.info(f'GNN proxy : {loss_proxies:.2f}')
                    self.logger.info(f'Total loss: {loss:.2f}')

            loss.backward()
            optimizer.step()
            first_batch = False
    '''
    def test_run(self, model: nn.Module, dl_ev: DataLoader, plots_dir, model_gnn=None):
        with torch.no_grad():
            recalls, nmi = self.evaluator.evaluate(
                model,
                dl_ev,
                dataroot=self.config['dataset']['name'],
                num_classes=self.config['dataset']['train_classes'],
                tsne=True,
                plot_dir=plots_dir
            )
            return recalls, nmi

    def get_logger(self):
        logger = logging.getLogger('Trainer')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def get_dataloaders_ssl(
        self,
        data_path: str,
        train_classes: int,
        labeled_fraction: float,
        num_workers: int
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader], DataLoader]:
        trans_train, trans_train_strong, trans_eval = get_transforms(
            self.config['dataset']['transform_ulb_strong'],
            self.config['dataset']['random_erasing'],
            self.config['dataset']['randaugment_num_ops'],
            self.config['dataset']['randaugment_magnitude']
        )
        self.logger.info('Transform (train_weak, train_strong, eval):\n{}\n{}\n{}'.format(
            trans_train,
            trans_train_strong,
            trans_eval
        ))

        dset_lb, dset_ulb, dset_eval = create_datasets(
            osp.join(data_path, 'images'),
            train_classes,
            labeled_fraction,
            trans_train,
            trans_train_strong,
            trans_eval,
            self.config['dataset']['transform_lb_strong']
        )
        self.logger.info('{} train samples ({} labeled + {} unlabeled)'.format(
            len(dset_lb) + len(dset_ulb),
            len(dset_lb),
            len(dset_ulb))
        )
        self.logger.info(f'{len(dset_eval)} eval samples')

        class_per_batch = self.config['training']['num_classes_iter']
        elements_per_class = self.config['training']['num_elements_class']

        batch_size_lb = class_per_batch * elements_per_class
        batch_size_ulb = self.config['training']['ulb_batch_size_factor'] * batch_size_lb

        if not self.labeled_only:
            num_batches = max(len(dset_ulb) // batch_size_ulb, len(dset_lb) // batch_size_lb)
        else:
            num_batches = len(dset_lb) // batch_size_lb

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
                pin_memory=True,
            )
            if not self.labeled_only:
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
                    pin_memory=True,
                )
            self.logger.info(f'Batch size labeled: {batch_size_lb}')
            self.logger.info(f'Batch size unlabeled: {batch_size_ulb}')
            self.logger.info(f'Num batches: {num_batches}')
            if self.labeled_only:
                assert len(dl_train_lb) == num_batches
            else:
                assert dl_train_ulb and len(dl_train_lb) == len(dl_train_ulb) == num_batches

        dl_eval = DataLoader(
            dset_eval,
            batch_size=batch_size_lb,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
        return dl_train_lb, dl_train_ulb, dl_eval

    def sample_hypers(self):
        random.seed()
        if self.config['dataset']['name'] != 'SOP':
            num_classes_iter, num_elements_class = random.choice([
                (8, 8), (6, 10), (5, 12), (4, 16), (3, 20), (10, 6), (12, 5), (16, 4), (20, 3)])
        else:
            num_classes_iter, num_elements_class = random.choice([(10, 6), (12, 5), (16, 4), (20, 3)])
        train_config = {
            'epochs': 40,
            'lr': 10 ** random.uniform(-5, -3),
            'weight_decay': 10 ** random.uniform(-15, -6),
            'num_classes_iter': num_classes_iter,
            'num_elements_class': num_elements_class,
            'loss_lb_temp': random.random(),
            'loss_ulb_weight': random.choice([1, 2, 5, 10]),
            'loss_ulb_threshold': random.choice([0.8, 0.85, 0.9]),
            'loss_ulb_gnn_threshold': random.choice([0.8, 0.85, 0.9]),
            'loss_proxy': random.choice([True, False])
        }
        self.config['training'].update(train_config)

        dataset_config = {
            'randaugment_num_ops': random.randint(2, 4),
            'randaugment_magnitude': random.randint(5, 15),
        }
        self.config['dataset'].update(dataset_config)

        gnn_config = {
            #'num_proxies': random.choice([
            #    self.config['dataset']['train_classes'],
            #    2 * self.config['dataset']['train_classes'],
            #    self.config['dataset']['train_classes'] // 2
            #]),
            'num_heads': random.choice([1, 2, 4]),
            'num_layers': random.choice([1, 2]),
            'add_mlp': random.choice([True, False]),
            'gnn_conv': random.choice(['GAT', 'MDP']),
            'gnn_fc': random.choice([True, False])
        }
        self.config['gnn'].update(gnn_config)

    def reduce_lr(self, model: nn.Module, optimizer: Optimizer):
        self.logger.info('Reducing learning rate:')
        path = osp.join(self.results_dir, self.filename)
        model.load_state_dict(torch.load(path))
        for g in optimizer.param_groups:
            self.logger.info(f'{g["lr"]} -> {g["lr"] / 10}')
            g['lr'] /= 10.


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = f"{seed}"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
