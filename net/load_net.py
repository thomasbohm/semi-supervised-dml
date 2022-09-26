import torch
import torch.nn as nn
from .utils import weights_init_kaiming, weights_init_classifier
from .resnet import resnet50


def load_resnet50(num_classes, pretrained_path='no', reduction=4, last_stride=0, neck=False):
    model = resnet50(
        pretrained=True,
        last_stride=last_stride,
        neck=neck,
        red=reduction
    )
    embed_dim = int(2048 / reduction)

    if neck:
        model.bottleneck = nn.BatchNorm1d(embed_dim)
        model.bottleneck.bias.requires_grad_(False)  # no shift
        model.fc = nn.Linear(embed_dim, num_classes, bias=False)

        model.bottleneck.apply(weights_init_kaiming)
        model.fc.apply(weights_init_classifier)
    else:
        model.fc = nn.Linear(embed_dim, num_classes)

    if pretrained_path not in ['', 'no']:
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(
                pretrained_path, map_location=torch.device('cpu')))
        else:
            state_dict = torch.load(pretrained_path)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:7] != 'module.':
                    new_state_dict = state_dict
                    break
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)
        print('Loaded weights: {}'.format(pretrained_path))

    return model, embed_dim
