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


class CategoricalPolicy(nn.Module):
    """
        Outputs the probability of actions
    """

    def __init__(self, state_size: int, act_dim: int, hidden_size: list = [1000, 1000, 1000], **args):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size[0])

        net_layers = mlp(hidden_size[0], hidden_layers=hidden_size[1:] + [act_dim],
                         activation=nn.ReLU, size=2, output_activation=nn.Sigmoid, as_list=True)

        self.logits = nn.Sequential(self.fc1, *net_layers[:])

        def forward(self, obs, act=None):
            """
                Gives a new policy under the
                observatation.

                If `act` is present, return the log
                probabilities of the actions under
                the new policy
            """
            pi = self.sample_policy(obs)
            log_p = None

            if act is not None:
                log_p = self.log_p(pi, act.unsqueeze(-1))

            return pi, log_p

        def sample_policy(self, obs):
            """
                Returns a new policy on the given observations
            """

            act_probs = self.logits(obs)

            pi = torch.distributions.Categorical(probs=act_probs)

        @classmethod
        def log_p(cls, pi, a):
            """
                Log probabilies of act w.r.t pi
            """

            return pi.log_p(a)

        def step(self, obs, log_p: bool = True):
            """
                Predict action
            """

            log_p = None
            with torch.no_grad():
                pi = self.sample_policy(obs)
                a = pi.sample()

                log_p = self.log_p(pi, a)

            return a.cpu().numpy(), log_p.cpu().numpy()


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

        def predict_v(self, obs):
            """
                Predicts the value of a state
            """

            with torch.no_grad():
                return self(obs).cpu().numpy()
