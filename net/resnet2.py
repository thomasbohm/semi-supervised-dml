import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init
from torchvision.models import resnet50
import torch.utils.checkpoint as checkpoint


class Resnet50(nn.Module):
    def __init__(self, embedding_size, num_classes, pretrained=True, is_norm=True, bn_freeze=True):
        super().__init__()
        #weights = "IMAGENET1K_V2" if pretrained else None
        self.model = resnet50(pretrained=pretrained)
        self.is_norm = is_norm
        self.embedding_size = embedding_size
        self.num_ftrs = self.model.fc.in_features
        self.model.gap = nn.AdaptiveAvgPool2d(1)
        self.model.gmp = nn.AdaptiveMaxPool2d(1)

        self.model.embedding = nn.Linear(self.num_ftrs, self.embedding_size)
        self.model.fc2 = nn.Linear(self.embedding_size, num_classes)
        self._initialize_weights()

        if bn_freeze:
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad_(False)
                    m.bias.requires_grad_(False)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x, output_option=None, val=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = checkpoint.checkpoint(self.model.layer1, x)
        x = checkpoint.checkpoint(self.model.layer2, x)
        x = checkpoint.checkpoint(self.model.layer3, x)
        x = checkpoint.checkpoint(self.model.layer4, x)

        avg_x = self.model.gap(x)
        max_x = self.model.gmp(x)
        x = max_x + avg_x

        x = x.view(x.size(0), -1)
        x = self.model.embedding(x)
        preds = self.model.fc2(x)
        
        if self.is_norm:
            x = self.l2_norm(x)
        
        return preds, x

    def _initialize_weights(self):
        init.kaiming_normal_(self.model.embedding.weight, mode='fan_out')
        init.constant_(self.model.embedding.bias, 0)
        init.kaiming_normal_(self.model.fc2.weight, mode='fan_out')
        init.constant_(self.model.fc2.bias, 0)
