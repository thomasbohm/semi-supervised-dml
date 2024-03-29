import math
import os
import random
from collections import defaultdict

from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from dataset.transforms import RandomErasing


def create_datasets(
    data_path,
    train_classes,
    labeled_fraction,
    transform_train,
    transform_train_strong,
    transform_eval,
    train_lb_uses_strong
):
    data_lb, data_ulb, data_eval = split_dataset(data_path,
                                                 range(train_classes),
                                                 labeled_fraction)

    assert len(data_lb) + len(data_ulb) + \
        len(data_eval) == len(ImageFolder(data_path))

    if not train_lb_uses_strong:
        dset_lb = SSLDataset(data_lb, transform=transform_train, is_ulb=False)
    else:
        dset_lb = SSLDataset(data_lb, transform=transform_train_strong, is_ulb=False)
    dset_ulb = SSLDataset(
        data_ulb,
        transform=transform_train,
        is_ulb=True,
        strong_transform=transform_train_strong
    )
    dset_eval = SSLDataset(data_eval, transform=transform_eval, is_ulb=False)

    return dset_lb, dset_ulb, dset_eval


def split_dataset(root, train_labels, labeled_fraction):
    dset = ImageFolder(root=root)
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


def get_transforms(transform_ulb_strong: str, random_erasing: bool, randaugment_num_ops: int, randaugment_magnitude: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    sz_resize = 256
    sz_crop = 227
    normalize_transform = transforms.Normalize(mean=mean, std=std)

    transform_train = [
        transforms.RandomResizedCrop(sz_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_transform
    ]

    if transform_ulb_strong == 'randaugment':
        transform_train_strong = [
            transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform
        ]
    elif transform_ulb_strong == 'simclr':
        transform_train_strong = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.8,
                    contrast=0.8,
                    saturation=0.8,
                    hue=0.2
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(23),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    else:
        raise NotImplementedError(f'Strong transform not supported: "{transform_ulb_strong}"')

    if random_erasing:
        re_transform = RandomErasing(
            probability=0.5,
            mean=(0.4914, 0.4822, 0.4465)
        )
        transform_train.append(re_transform)
        transform_train_strong.append(re_transform)

    transform_train = transforms.Compose(transform_train)
    transform_train_strong = transforms.Compose(transform_train_strong)

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
        If is labeled dataset,
            return weak_augment_image, target, path
        else:
            return weak_augment_image, strong_augment_image, target, path
        """
        img, target = self.data[idx]
        transformed = self.transform(img) if self.transform else img
        path = self.data.dataset.samples[self.data.indices[idx]][0]

        if not self.is_ulb:
            return transformed, target, path
        else:
            return transformed, self.strong_transform(img), target, path

    def __len__(self):
        return len(self.targets)
