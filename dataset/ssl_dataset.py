import os
import math
import random

from collections import defaultdict
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Subset, Dataset


def create_datasets(
        data_path,
        train_classes,
        labeled_fraction,
        transform_train,
        transform_train_strong,
        transform_eval
    ):
    data_lb, data_ulb, data_eval = split_dataset(data_path,
                                                 range(train_classes),
                                                 labeled_fraction)
    
    assert len(data_lb) + len(data_ulb) + len(data_eval) == len(ImageFolder(data_path))
    
    dset_lb = SSLDataset(data_lb, transform=transform_train, is_ulb=False)
    dset_ulb = SSLDataset(data_ulb,
                          transform=transform_train,
                          is_ulb=True,
                          strong_transform=transform_train_strong)
    dset_eval = SSLDataset(data_eval, transform=transform_eval, is_ulb=False)

    return dset_lb, dset_ulb, dset_eval

def split_dataset(root, train_labels, labeled_fraction):    
    dset = ImageFolder(root=os.path.join(root, 'images'))
    cls_to_indices = defaultdict(list)

    eval_indices = []
    for idx, (img, label) in enumerate(dset.imgs):
        if label in train_labels:
            cls_to_indices[label].append(idx)
        else:
            eval_indices.append(idx)

    lb_indices, ulb_indices = [], []
    
    for label, indices in cls_to_indices.items():
        num_labeled = math.ceil(labeled_fraction * len(indices))
        random.shuffle(indices)
        
        labeled, unlabeled = indices[:num_labeled], indices[num_labeled:]
        lb_indices += labeled
        ulb_indices += unlabeled
    
    lb_dset, ulb_dset = Subset(dset, lb_indices), Subset(dset, ulb_indices)
    eval_dset = Subset(dset, eval_indices)
    
    return lb_dset, ulb_dset, eval_dset


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std  = [0.299, 0.224, 0.225]
    sz_resize = 256
    sz_crop = 227
    normalize_transform = transforms.Normalize(mean=mean, std=std)
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(sz_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform
    ])
    transform_train_strong = transforms.Compose([
        transforms.RandAugment(),
        transforms.RandomResizedCrop(sz_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform
    ])
    transform_eval = transforms.Compose([
        transforms.Resize(sz_resize),
        transforms.CenterCrop(sz_crop),
        transforms.ToTensor(),
        normalize_transform
    ])
    return transform_train, transform_train_strong, transform_eval


class SSLDataset(Dataset):
    def __init__(self, data, transform=None, is_ulb=False, strong_transform=None):
        self.data = data
        self.is_ulb = is_ulb
        self.transform = transform
        self.targets = [d[1] for d in data]
        if self.is_ulb:
            assert strong_transform is not None
            self.strong_transform = strong_transform

                
    def __getitem__(self, idx):
        """
        If labeled datset,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target = self.data[idx]
        transformed = self.transform(img) if self.transform else img
        
        if not self.is_ulb:
            return transformed, target
        else:
            return transformed, self.strong_transform(img), target
            
            
    def __len__(self):
        return len(self.targets)
