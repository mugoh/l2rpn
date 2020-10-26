"""
    Actor and Critic Modules
"""
import typing

import torch
import torch.nn as nn


import numpy as np
import scipy.signal


def disc_cumsum(x: typing.List, disc: int) -> int:
    """
        Returns a time decayed discounted cumulative
        sum of `x` with `disc` as the discounting factor
    """

    return scipy.signal.lfilter([1], [1, float(-disc)], x[::-1],
                                axis=0)[::-1]


def mlp(x,
        hidden_layers,
        activation=nn.Tanh,
        size=2,
        output_activation=nn.Identity, as_list=False):
    """
        Multi-layer perceptron
    """
    net_layers = []

    if len(hidden_layers[:-1]) < size:
        hidden_layers[:-1] *= size

    for size in hidden_layers[:-1]:
        layer = nn.Linear(x, size)
        net_layers.append(layer)
        net_layers.append(activation())
        x = size

    net_layers.append(nn.Linear(x, hidden_layers[-1]))
    net_layers += [output_activation()]

    if as_list:
        return net_layers
    return nn.Sequential(*net_layers)


class Actor(nn.Module):
    """
        Outputs the probability of each action
    """

    def __init__(self, state_size: int, act_dim: int, hidden_size: list = [1000, 1000, 1000], **args):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size[0])

        net_layers = mlp(hidden_size[0], hidden_layers=hidden_size[1:] + [act_dim],
                         activation=nn.ReLU, size=2, output_activation=nn.Sigmoid, as_list=True)

        self.act_prob = nn.Sequential(self.fc1, *net_layers[:])

        def forward(self, x):
            return self.act_prob(x)


class Critic(nn.Module):
    """
        Predicts the value of each state
    """

    def __init__(self, shared_layer, state_size: int, act_dim: int, hidden_size: list = [1000, 1000, 1000], **args):
        super(Critic, self).__init__()

        self.fc1 = shared_layer

        net_layers = mlp(hidden_size[0], hidden_layers=hidden_size[1:] + [1],
                         activation=nn.ReLU, size=2, output_activation=nn.Identity, as_list=True)

        self.value_f = nn.Sequential(self.fc1, *net_layers[:])

        def forward(self, x):
            return self.value_f(x).squeeze(-1)
