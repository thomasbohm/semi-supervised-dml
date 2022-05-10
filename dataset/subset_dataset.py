import os
import math
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .utils import pil_loader

class SubsetDataset(Dataset):
    def __init__(self, root, included_labels, labeled_fraction, transform):        
        self.included_labels = included_labels
        self.transform = transform
        
        cls_to_paths = defaultdict(list)
        imgfolder = ImageFolder(root=os.path.join(root, 'images'))
        
        for img, label in imgfolder.imgs:
            if label in included_labels:
                cls_to_paths[label].append(img)
        
        self.img_paths, self.targets = [], []
        # self.img_paths_unlabeled, self.targets_unlabeled = [], []

        for label, paths in cls_to_paths.items():
            num_labeled = math.ceil(labeled_fraction * len(paths))
            random.shuffle(paths)
            labeled, unlabeled = paths[:num_labeled], paths[num_labeled:]
            self.img_paths += labeled
            self.targets += [label] * num_labeled
            # self.img_paths_unlabeled += unlabeled
            # self.targets_unlabeled += [label] * (len(paths) - num_labeled)
    
    
    def num_classes(self):
        n = len(np.unique(self.targets))
        assert n == len(self.included_labels)
        return n

    
    def __len__(self):
        return len(self.targets)

    
    def __getitem__(self, index):
        img = pil_loader(self.img_paths[index])
        target = self.targets[index]

        if self.transform:
            img = self.transform(img)

        return img, target
