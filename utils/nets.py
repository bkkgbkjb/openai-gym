from copy import deepcopy
from torch import nn
import torch
import numpy as np


class NeuralNetworks(nn.Module):
    def __init__(self):
        super(NeuralNetworks, self).__init__()

    def hard_update_to(self, target: nn.Module):
        # assert self.parameters() == len(target.parameters())
        for c, t in zip(self.parameters(), target.parameters()):
            c.data.copy_(t.data)
        return self

    def soft_update_to(self, target: nn.Module, tau: float):
        for c, t in zip(self.parameters(), target.parameters()):
            c.data.copy_(c.data * (1 - tau) + t.data * tau)
        return self

    def no_grad(self):
        for p in self.parameters():
            p.requires_grad = False
        return self

    def clone(self):
        return deepcopy(self)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
