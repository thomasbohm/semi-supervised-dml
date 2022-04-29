import torch
import torch.nn as nn
from .utils import weights_init_kaiming, weights_init_classifier
from .resnet import resnet50

def load_net(red, num_classes, pretrained_path, last_stride=0, neck=0):
    sz_embed = int(2048/red)
    model = resnet50(pretrained=True, last_stride=last_stride, neck=neck, red=red)

    dim = int(2048/red)
    if neck:
        model.bottleneck = nn.BatchNorm1d(dim)
        model.bottleneck.bias.requires_grad_(False)  # no shift
        model.fc = nn.Linear(dim, num_classes, bias=False)

        model.bottleneck.apply(weights_init_kaiming)
        model.fc.apply(weights_init_classifier)
    else: 
        model.fc = nn.Linear(dim, num_classes)

    if pretrained_path != 'no':
        if not torch.cuda.is_available(): 
            model.load_state_dict(torch.load(pretrained_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(pretrained_path))
    
    return model, sz_embed