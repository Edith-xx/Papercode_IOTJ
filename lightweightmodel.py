import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from data_loader import *
class Bottle2neck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, scale=3):
        super(Bottle2neck, self).__init__()
        self.groups = planes//scale
        self.len = planes//scale
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)

        self.conv2 = nn.Conv2d(planes//scale, planes//scale, kernel_size=3, groups=planes//scale, padding=1)
        self.conv3 = nn.Conv2d(planes//scale, planes//scale, kernel_size=1, stride=1)
        self.scale = scale
        self.shortcut = None
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride),
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        #out =shuffle_chnls(out, self.groups)
        spx = torch.split(out, self.len, 1) #从通道的维度切割,待分输入，需要切分的大小，切分维度
        side = self.conv2(spx[1])
        side = self.relu(side)
        side = self.conv3(side)
        z = torch.cat((spx[0], side),1)
        for i in range(2, self.scale):
            sp = side + spx[i]
            y = self.conv2(sp)
            y = self.relu(y)
            y = self.conv3(y)
            side = y
            z = torch.cat((z,y),1)
        out = z
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = out + residual
        out = self.relu(out)
        return out

class Res2Net(nn.Module):

    def __init__(self, block, layers, scale, num_classes):
        self.inplanes = 32
        super(Res2Net, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=(54,25))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 9, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 18, layers[1], stride=1)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.LazyLinear(num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride, scale=self.scale))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, scale=self.scale))
        return nn.Sequential(*layers)
    def forward(self, x, snr):

        x = self.conv(x)
        x = self.relu(x)
        x = self.layer1(x)
        features = x
        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x1 = torch.cat((x, snr), 1)
        x = self.fc(x1)
        return x
def my_resnet():
    model = Res2Net(Bottle2neck, [2, 2], 3, 7)
    return model

'''
shuffle_chnls
def shuffle_chnls(x, groups=2):
    """Channel Shuffle"""

    bs, chnls, h, w = x.data.size()
    if chnls % groups:
        return x
    chnls_per_group = chnls // groups
    x = x.view(bs, groups, chnls_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bs, -1, h, w)
    return x
'''