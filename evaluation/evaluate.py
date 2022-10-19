import os.path as osp
from typing import List, Tuple
import torch
import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch.nn.functional as F
from .normalized_mutual_information import calc_normalized_mutual_information, cluster_by_kmeans
from .recall import calc_recall_at_k, assign_by_euclidian_at_k


class Evaluator():
    def __init__(self, device, logging_level=logging.INFO):
        self.device = device

        self.logger = logging.getLogger('Evaluator')
        self.logger.setLevel(logging_level)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.tsne_model = TSNE(n_components=2, learning_rate='auto', random_state=0, init='pca')

    @torch.no_grad()
    def evaluate(
        self,
        model,
        dataloader,
        dataroot,
        num_classes,
        tsne=False,
        plot_dir='',
    ) -> Tuple[List[float], float]:
        model_is_training = model.training
        model.eval()

        feats, targets = self._predict_batchwise(model, dataloader)

        if tsne:
            self._create_tsne_plot(feats, targets, osp.join(plot_dir, 'tsne_backbone.svg'))

        if dataroot != 'SOP':
            Y, targets = assign_by_euclidian_at_k(feats, targets, 8)
            which_nearest_neighbors = [1, 2, 4, 8]
        else:
            Y, targets = assign_by_euclidian_at_k(feats, targets, 1000)
            which_nearest_neighbors = [1, 10, 100, 1000]

        recalls: List[float] = []
        for k in which_nearest_neighbors:
            r_at_k = calc_recall_at_k(targets, Y, k)
            recalls.append(r_at_k)
            self.logger.info("R@{}: {:.3f}".format(k, 100 * r_at_k))

        if dataroot != 'SOP':
            nmi = calc_normalized_mutual_information(targets, cluster_by_kmeans(feats, num_classes))
            self.logger.info("NMI: {:.3f}".format(nmi * 100))
        else:
            nmi = -1.0

        model.train(model_is_training)
        return recalls, nmi
    
    @torch.no_grad()
    def create_train_plots(self, backbone, gnn, dl_tr_lb, dl_tr_ulb, num_classes, plot_dir):
        x_lb, x_ulb_w, x_ulb_s, y = self._predict_batchwise_train(backbone, gnn, dl_tr_lb, dl_tr_ulb)
        x = torch.cat((x_lb, x_ulb_w, x_ulb_s))
        proxies = F.normalize(gnn.proxies, p=2, dim=1).cpu()
        
        num_proxies = proxies.shape[0]
        self._create_tsne_plot_gnn(
            torch.cat([proxies, x]),
            y,
            osp.join(plot_dir, 'tsne_gnn.svg'),
            num_proxies=num_proxies
        )
        self._create_distance_plot_gnn(
            x,
            y,
            proxies,
            num_classes,
            osp.join(plot_dir, 'dist_gnn.svg')
        )
    
    @torch.no_grad()
    def _predict_batchwise(self, model, dataloader):
        fc7s, targets = [], []
        for x, y, p in dataloader:
            x = x.to(self.device)
            try:
                preds, embeds = model(x, output_option='norm', val=True)
                fc7s.append(F.normalize(embeds, p=2, dim=1).cpu())
                targets.append(y)

            except TypeError:
                if torch.cuda.device_count() > 1:
                    # Pass this error.
                    # Happens if len(dset_eval) % batch_size is small
                    # and multi-gpu training is used. The last batch probably
                    # cannot be distributed onto all gpus.
                    self.logger.info(f'Skipping batch of shape {x.shape}')
                    pass
                else:
                    raise TypeError()

        fc7, targets = torch.cat(fc7s), torch.cat(targets)
        return torch.squeeze(fc7), torch.squeeze(targets)


    @torch.no_grad()
    def _predict_batchwise_train(self, backbone, gnn, dl_tr_lb, dl_tr_ulb):
        backbone.eval()
        gnn.eval()
        
        targets = []
        feats_gnn_lb = []
        feats_gnn_ulb_w = []
        feats_gnn_ulb_s = []
        for (x_lb, y_lb, p_lb), (x_ulb_w, x_ulb_s, y_ulb, p_ulb) in zip(dl_tr_lb, dl_tr_ulb):
            x = torch.cat((x_lb, x_ulb_w, x_ulb_s)).to(self.device)
            try:
                _, embeds = backbone(x, output_option='norm', val=True)

                torch.use_deterministic_algorithms(False)
                _, embeds_gnn = gnn(embeds)
                torch.use_deterministic_algorithms(True)
                
                embeds_gnn = F.normalize(embeds_gnn, p=2, dim=1).cpu()
                embeds_gnn_lb = embeds_gnn[:x_lb.shape[0]]
                embeds_gnn_ulb_w = embeds_gnn[x_lb.shape[0]:x_lb.shape[0] + x_ulb_w.shape[0]]
                embeds_gnn_ulb_s = embeds_gnn[x_lb.shape[0] + x_ulb_w.shape[0]:]

                feats_gnn_lb.append(embeds_gnn_lb)
                feats_gnn_ulb_w.append(embeds_gnn_ulb_w)
                feats_gnn_ulb_s.append(embeds_gnn_ulb_s)
                targets.append(torch.cat(y_lb, y_ulb, y_ulb))

            except TypeError:
                if torch.cuda.device_count() > 1:
                    # Pass this error.
                    # Happens if len(dset_eval) % batch_size is small
                    # and multi-gpu training is used. The last batch probably
                    # cannot be distributed onto all gpus.
                    self.logger.info(f'Skipping batch of shape {x.shape}')
                    pass
                else:
                    raise TypeError()

        targets = torch.squeeze(torch.cat(targets))
        feats_gnn_lb = torch.squeeze(torch.cat(feats_gnn_lb))
        feats_gnn_ulb_w = torch.squeeze(torch.cat(feats_gnn_ulb_w))
        feats_gnn_ulb_s = torch.squeeze(torch.cat(feats_gnn_ulb_s))
        return feats_gnn_lb, feats_gnn_ulb_w, feats_gnn_ulb_s, targets

    def _get_colors(self, Y: torch.Tensor):
        assert len(Y.shape) == 1
        colors = []
        color_map = {}
        for i in range(Y.shape[0]):
            y = Y[i].item()
            if y not in color_map:
                num_colors = len(color_map)
                color_map[y] = num_colors
            colors.append(color_map[y])
        return colors
    
    def _create_tsne_plot(self, feats, targets, path):
        with torch.no_grad():
            self.logger.info('Creating tsne embeddings...')
            feats_tsne = self.tsne_model.fit_transform(feats.detach().cpu())
            fig, ax = plt.subplots()
            ax.scatter(*feats_tsne.T, c=self._get_colors(targets), s=5, alpha=0.6, cmap='tab20')
            
            fig.set_size_inches(11.69,8.27)
            fig.savefig(path)
            self.logger.info(f'Saved plot to {path}')
    
    def _create_tsne_plot_gnn(self, feats, targets, path, num_proxies):
        with torch.no_grad():
            self.logger.info('Creating tsne gnn embeddings...')
            feats_tsne = self.tsne_model.fit_transform(feats.detach().cpu())
            fig, ax = plt.subplots()
            ax.scatter(*feats_tsne[num_proxies:].T, c=self._get_colors(targets), s=5, alpha=0.6, cmap='tab20')
            ax.scatter(*feats_tsne[:num_proxies].T, c=list(range(num_proxies)), s=50, alpha=1, marker='*', cmap='tab20')

            fig.set_size_inches(11.69,8.27)
            fig.savefig(path)
            self.logger.info(f'Saved plot to {path}')
    
    def _create_distance_plot_gnn(self, feats, targets, proxies, num_classes, path):
        data = self._get_proxies_to_class_avg(feats, proxies, targets, num_classes)

        fig, ax = plt.subplots()
        im = ax.imshow(data)

        ax.set_xlabel('Classes')
        ax.set_ylabel('Proxies')
        cbar = ax.figure.colorbar(im, ax=ax)
        fig.set_size_inches(11.69,8.27)
        fig.savefig(path)
        self.logger.info(f'Saved plot to {path}')

    def _get_proxies_to_class_avg(self, feats, proxies, targets, num_classes):
        dist = (feats[:, None, :] - proxies[None, :, :]) ** 2 # (N, 1, D) - (1, P, D)
        dist = dist.sum(dim=2) # (N, P)
        dist = torch.sqrt(dist)

        res = torch.zeros((num_classes, proxies.shape[0]))
        for cls in range(num_classes):
            res[cls] = dist[targets == cls].mean(dim=0)
        
        return res # (C, P)
