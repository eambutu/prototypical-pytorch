# coding=utf-8
import torch
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn.modules.loss import _assert_no_grad


class PrototypicalLoss(Module):
    '''
    Class à la PyTorch for the prototypical loss function defined below
    '''

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        _assert_no_grad(target)
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def obtain_mean(input, target, n_support):
    cputargs = target.cpu() if target.is_cuda else target
    cputargs = cputargs.data
    cpuinput = input.cpu() if target.is_cuda else input

    def supp_idxs(c):
        return torch.nonzero(cputargs.eq(int(c)))[:n_support].squeeze()

    classes = np.unique(cputargs)
    os_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([cpuinput[i].mean(0) for i in os_idxs])
    return prototypes


def prototypical_loss(input, prototypes, target, n_support, inner_loop=False):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    cputargs = target.cpu() if target.is_cuda else target
    cputargs = cputargs.data
    classes = np.unique(cputargs)
    n_classes = len(classes)
    n_query = len(torch.nonzero(cputargs.eq(int(classes[0])))) - n_support

    prototypes = prototypes.cuda() if target.is_cuda else prototypes
    if inner_loop:
        oq_idxs_0 = torch.stack(list(map(lambda c: torch.nonzero(cputargs.eq(int(c)))[:n_support], classes))).view(-1)
    else:
        oq_idxs_0 = torch.stack(list(map(lambda c: torch.nonzero(cputargs.eq(int(c)))[n_support:], classes))).view(-1)
    oq_idxs_0 = oq_idxs_0.cuda() if target.is_cuda else oq_idxs_0
    oq = input[oq_idxs_0]
    dists = euclidean_dist(oq, prototypes)

    if inner_loop:
        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_support, -1)
    else:
        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    if inner_loop:
        target_inds = target_inds.expand(n_classes, n_support, 1).long()
    else:
        target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds = target_inds.long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda() if target.is_cuda else target_inds

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)

    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    return loss_val,  acc_val

def full_loss(input, target, n_support, inner_loop=False):
    prototypes = obtain_mean(input, target, n_support)
    return prototypical_loss(input, prototypes, target, n_support, inner_loop)
