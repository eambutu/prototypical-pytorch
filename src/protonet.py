import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def batchnorm(input, weight=None, bias=None, running_mean=None, running_var=None, training=True,eps=1e-5, momentum=0.1):
    # momentum = 1 restricts stats to the current mini-batch
    # This hack only works when momentum is 1 and avoids needing to track
    # running stats by substituting dummy variables
    size = int(np.prod(np.array(input.data.size()[1])))
    running_mean = torch.zeros(size).cuda()
    running_var = torch.ones(size).cuda()
    return F.batch_norm(input, running_mean, running_var, weight, bias, training, momentum, eps)

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(in_channels, out_channels, 3, padding=1)),
        ('bn', nn.BatchNorm2d(out_channels, momentum=1)),
        #('bn', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('block1', conv_block(x_dim, hid_dim)),
            ('block2', conv_block(hid_dim, hid_dim)),
            ('block3', conv_block(hid_dim, hid_dim)),
            ('block4', conv_block(hid_dim, z_dim)),
        ]))

    def forward(self, x, weights=None):
        if weights is None:
            x = self.encoder(x)
        else:
            x = F.conv2d(x, weights['encoder.block1.conv.weight'], weights['encoder.block1.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block1.bn.weight'], bias=weights['encoder.block1.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block2.conv.weight'], weights['encoder.block2.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block2.bn.weight'], bias=weights['encoder.block2.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block3.conv.weight'], weights['encoder.block3.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block3.bn.weight'], bias=weights['encoder.block3.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
            x = F.conv2d(x, weights['encoder.block4.conv.weight'], weights['encoder.block4.conv.bias'])
            x = batchnorm(x, weight=weights['encoder.block4.bn.weight'], bias=weights['encoder.block4.bn.bias'])
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)
        return x.view(x.size(0), -1)
