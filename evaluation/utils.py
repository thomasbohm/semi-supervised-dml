import torch
import logging

from .normalized_mutual_information import calc_normalized_mutual_information, cluster_by_kmeans
from .recall import calc_recall_at_k, assign_by_euclidian_at_k


class Evaluator():
    def __init__(self, device, cat=0, logging_level=logging.INFO):
        self.cat = cat
        self.device = device

        self.logger = logging.getLogger('Evaluator')
        self.logger.setLevel(logging_level)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def evaluate(self, model, dataloader, dataroot, num_classes):
        model_is_training = model.training
        model.eval()

        feats, targets = self.predict_batchwise(model, dataloader)

        recalls = []
        if dataroot != 'SOP':
            Y, targets = assign_by_euclidian_at_k(feats, targets, 8)
            which_nearest_neighbors = [1, 2, 4, 8]
        else:
            Y, targets = assign_by_euclidian_at_k(feats, targets, 1000)
            which_nearest_neighbors = [1, 10, 100, 1000]

        for k in which_nearest_neighbors:
            r_at_k = calc_recall_at_k(targets, Y, k)
            recalls.append(r_at_k)
            self.logger.info("R@{}: {:.3f}".format(k, 100 * r_at_k))

        if dataroot != 'SOP':
            nmi = calc_normalized_mutual_information(
                targets, cluster_by_kmeans(feats, num_classes))
            self.logger.info("NMI: {:.3f}".format(nmi * 100))
        else:
            nmi = -1

        model.train(model_is_training)
        return recalls, nmi

    def predict_batchwise(self, model, dataloader):
        fc7s, targets = [], []
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                try:
                    _, fc7 = model(x, output_option='plain', val=True)
                    fc7s.append(fc7.cpu())
                    targets.append(y)
                except TypeError:
                    if torch.cuda.device_count() > 1:
                        # Silenty skip this error.
                        # Happens if len(dset_eval) % batch_size is small
                        # and multi-gpu training is used. The last batch probably
                        # cannot be distributed onto all gpus.
                        self.logger.info(f'Skipping batch of shape {x.shape}')
                        pass
                    else:
                        raise TypeError()

        fc7, targets = torch.cat(fc7s), torch.cat(targets)
        return torch.squeeze(fc7), torch.squeeze(targets)
