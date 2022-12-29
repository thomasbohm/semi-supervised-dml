import torch
from os import path as osp
from torch.utils.data import DataLoader
from collections import OrderedDict
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

from dataset.ssl_dataset import create_datasets, get_transforms
from net.load_net import load_resnet50


def load_model_and_data():
    data_path = "data/CARS"
    pretrained_path = "./results/CARS/2022-11-19_02:19:16/CARS_10.0_best.pth"
    train_classes = 98
    ra_num_ops = 4
    ra_magnitude = 5
    num_classes_iter = 12
    num_elements_class = 5

    trans_train, trans_train_strong, trans_eval = get_transforms(
        'randaugment',
        False, # RE
        ra_num_ops,
        ra_magnitude,
    )

    _, _, dset_eval = create_datasets(
        osp.join(data_path, 'images'),
        train_classes,
        0.1,
        trans_train,
        trans_train_strong,
        trans_eval,
        False
    )

    dl_eval = DataLoader(
        dset_eval,
        batch_size=num_classes_iter * num_elements_class,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    resnet, _ = load_resnet50(train_classes)

    state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] != 'module.':
            new_state_dict = state_dict
            break
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    resnet.load_state_dict(new_state_dict)

    return resnet, dl_eval

def predict_batchwise(model, dataloader, device):
        fc7s, targets, paths = [], [], []
        i = 1
        for x, y, p in dataloader:
            print(f'{i}/{len(dataloader)}')
            x = x.to(device)
            _, embeds = model(x, output_option='norm', val=True)
            fc7s.append(F.normalize(embeds, p=2, dim=1).cpu())
            targets.append(y)
            paths += p
            i += 1
        print()

        fc7, targets = torch.cat(fc7s), torch.cat(targets)
        return torch.squeeze(fc7), torch.squeeze(targets), paths

def print_closest(resnet, dl_eval, device):
    feats, targets, paths = predict_batchwise(resnet, dl_eval, device)
    print(f'feats.shape: {feats.shape}')
    print(f'len(paths): {len(paths)}')

    dist = torch.cdist(feats, feats)
    print(f'dist.shape: {dist.shape}')
    _, closest_idx = torch.topk(dist, k=5, dim=1, largest=False)    
    print(f'closest_idx.shape: {closest_idx.shape}')

    samples = random.sample(range(feats.shape[0]), 10)

    for row, sample in enumerate(samples, start=1):
        print(f'Sample {row}/10:')
        print((paths[sample], [paths[idx] for idx in closest_idx[sample]]))
        print('-' * 10)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet, dl_eval = load_model_and_data()
resnet = resnet.to(device)

print_closest(resnet, dl_eval, device)
