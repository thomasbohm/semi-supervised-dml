from torch.utils.data.sampler import Sampler
import random
import copy
import numpy as np
import logging

logger = logging.getLogger('GNNReID.CombineSampler')

class CombineSampler(Sampler):
    """
    l_inds (list of lists)
    cl_b (int): classes in a batch
    n_cl (int): num of obs per class inside the batch
    """

    def __init__(self, l_inds, cl_b, n_cl, batch_sampler=None):
        logger.info("Combine Sampler")
        self.l_inds = l_inds
        self.max = -1
        self.num_classes = cl_b
        self.num_elements_per_class = n_cl
        self.batch_size = cl_b * n_cl
        self.flat_list = []
        self.feature_dict = None
        for inds in l_inds:
            if len(inds) > self.max:
                self.max = len(inds)
        
        #if batch_sampler == 'NumberSampler':
        #    self.sampler = NumberSampler(cl_b, n_cl)
        #elif batch_sampler == 'BatchSizeSampler':
        #    self.sampler = BatchSizeSampler()
        #else:
        self.sampler = None

    def __iter__(self):
        if self.sampler:
            self.num_classes, self.num_elements_per_class = self.sampler.sample()

        # shuffle elements inside each class
        l_inds = list(map(lambda a: random.sample(a, len(a)), self.l_inds))

        for inds in l_inds:
            choose = copy.deepcopy(inds)
            while len(inds) < self.num_elements_per_class:
                inds += [random.choice(choose)]

        # split lists of a class every n_cl elements
        split_list_of_indices = []
        for inds in l_inds:
            inds = inds + np.random.choice(inds, size=(len(inds) // self.num_elements_per_class + 1)*self.num_elements_per_class - len(inds), replace=False).tolist()
            # drop the last < n_cl elements
            while len(inds) >= self.num_elements_per_class:
                split_list_of_indices.append(inds[:self.num_elements_per_class])
                inds = inds[self.num_elements_per_class:] 
            assert len(inds) == 0
        # shuffle the order of classes --> Could it be that same class appears twice in one batch?
        random.shuffle(split_list_of_indices)
        if len(split_list_of_indices) % self.num_classes != 0:
            b = np.random.choice(np.arange(len(split_list_of_indices)), size=self.num_classes - len(split_list_of_indices) % self.num_classes, replace=False).tolist()
            [split_list_of_indices.append(split_list_of_indices[m]) for m in b]
        
        self.flat_list = [item for sublist in split_list_of_indices for item in sublist]
        return iter(self.flat_list)

    def __len__(self):
        return len(self.flat_list)