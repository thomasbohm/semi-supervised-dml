from .normalized_mutual_information import calc_normalized_mutual_information, cluster_by_kmeans
from .recall import calc_recall_at_k, assign_by_euclidian_at_k

import torch
import logging
import sklearn.cluster
import sklearn.metrics.cluster
import time
import tqdm


class Evaluator_DML():
    def __init__(self, config, device, cat=0):
        self.cat = cat
        self.device = device

        logger = logging.getLogger('eval')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s:%(name)s: %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler('./results/{}_eval_{}.log'.format(config['dataset']['name'], time.time()))
        fh.setFormatter(formatter)
        
        logger.addHandler(fh)
        self.logger = logger
        self.logger.info('PARAMS:')
        self.logger.info('Dataset: {}'.format(config['dataset']['name']))
        self.logger.info('Labeled: {}'.format(config['dataset']['labeled_fraction']))
        self.logger.info('Epochs: {}'.format(config['training']['epochs']))
        self.logger.info('{}'.format('-' * 10))


    def evaluate(self, model, dataloader, dataroot, num_classes):
        self.num_classes = num_classes
        start = time.time()
        model_is_training = model.training
        model.eval()
        
        # calculate embeddings with model, also get labels (non-batch-wise)
        X, T, P = self.predict_batchwise(model, dataloader)
        
        if dataroot != 'in_shop' and dataroot != 'sop':
            # calculate NMI with kmeans clustering
            nmi = calc_normalized_mutual_information(T, cluster_by_kmeans(X, num_classes))
            self.logger.info("NMI: {:.3f}".format(nmi * 100))
        else:
            nmi = -1
        
        # Get Recall
        recall = []
        if dataroot != 'sop':
            Y, T = assign_by_euclidian_at_k(X, T, 8)
            which_nearest_neighbors = [1, 2, 4, 8]
        else:
            Y, T = assign_by_euclidian_at_k(X, T, 1000)
            which_nearest_neighbors = [1, 10, 100, 1000]
        
        for k in which_nearest_neighbors:
            r_at_k = calc_recall_at_k(T, Y, k)
            recall.append(r_at_k)
            self.logger.info("R@{} : {:.3f}".format(k, 100 * r_at_k))

        model.train(model_is_training)
        return nmi, recall

    # just looking at this gives me AIDS, fix it fool!
    def predict_batchwise(self, model, dataloader):
        # self.logger.info("Evaluate normal")
        paths = []
        fc7s, Ys = list(), list()
        with torch.no_grad():
            for X, Y in dataloader:
                X = X.to(self.device)
                _, fc7 = model(X, output_option='plain', val=True)
                
                fc7s.append(fc7)
                Ys.append(Y)
                #paths.append(P)
                
        fc7 = torch.cat([f.unsqueeze(0).cpu() for b in fc7s for f in b], 0)
        Y = torch.cat([y.unsqueeze(0).cpu() for b in Ys for y in b], 0)
        paths = [p for b in paths for p in b]
        
        return torch.squeeze(fc7), torch.squeeze(Y), paths

    def get_resnet_performance(self, x, ys):
        cluster = sklearn.cluster.KMeans(self.nb_classes).fit(x).labels_
        NMI = sklearn.metrics.cluster.normalized_mutual_info_score(cluster, ys)
        self.logger.info("KNN: NMI after ResNet50 {}".format(NMI))

        RI = sklearn.metrics.adjusted_rand_score(ys, cluster)
        self.logger.info("RI after Resnet50 {}".format(RI))

        Y, ys_ = assign_by_euclidian_at_k(x, ys, 1)
        r_at_k = calc_recall_at_k(ys_, Y, 1)
        self.logger.info("KNN: R@{} after ResNet50: {:.3f}".format(1, 100 * r_at_k))


def predict_batchwise(model, dataloader):
    model_is_training = model.training
    model.eval()

    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * paths
                if i == 0:
                    # move images to device of model (approximate device)
                    _, J = model(J.cuda())


                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state

    return torch.stack(A[0]), torch.stack(A[1]), A[2]


def get_recall(query_X, gallery_X, gallery_T, query_T):
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    import torch.nn.functional as F 
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1

        return match_counter / m

    # calculate recall @ 1, 2, 4, 8
    recall = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))
    quit()
    return recall


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output
