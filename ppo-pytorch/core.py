import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from gym.spaces import Discrete

import scipy.signal


def discounted_cumsum(rew, disc):
    """
        Calculates rewards to go
    """
    rew_len = len(rew)
    disc_array = np.repeat(disc, rew_len)

    all_rtg = []

    for t in range(rew_len):

        # indices t' to T-1
        indices = np.arange(t, rew_len)

        # gamma^(t'-t)
        discounts = np.power(disc, indices - t)

        # r_t' * decay
        rtg = rew[t:, ] * discounts

        # sum_{t'=t}^{T-1} rtg
        rtg_sum = np.sum(rtg)

        all_rtg.append(rtg_sum)

    return np.asanyarray(all_rtg)

def dcum2(rew, disc):
    """
        Compare against discounted_cumsum
        # Much faster than discounted_cumsum
    """
    return scipy.signal.lfilter([1], [1, float(-disc)], rew[::-1], axis=0)[::-1]

def count(module):
    """
        Returns a count of the parameters
        in a module
    """
    return np.sum([np.prod(p.shape) for p in module.parameters()])


def mlp(x, hidden_layers, activation=nn.Tanh, size=2, output_activation=nn.Identity):
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

    return nn.Sequential(*net_layers)


class MLPCritic(nn.Module):
    """
        Agent Critic
        Estmates value function
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, size=2):
        super(MLPCritic, self).__init__()

        self.net = mlp(obs_dim, hidden_sizes +
                       [1], activation=activation, size=size)

    def forward(self, obs):
        """
            Get value function estimate
        """
        return torch.squeeze(self.net(obs), axis=-1)


class Actor(nn.Module):
    def __init__(self, **args):
        super(Actor, self).__init__()

    def forward(self, obs, ac=None):
        """
            Gives policy for given observations
            and optionally actions log prob under that
            policy
        """
        pi = self.sample_action(obs)
        log_p = None

        if isinstance(self, CategoricalPolicy):
            ac=ac.unsqueeze(-1)

        if ac is not None:
            log_p = self.log_p(pi, ac)
        return pi, log_p


class MLPGaussianPolicy(Actor):
    """
        Gaussian Policy for stochastic actions
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, size=2):
        super(MLPGaussianPolicy, self).__init__()

        self.logits = mlp(obs_dim, hidden_sizes +
                          [act_dim], activation, size=size)
        log_std = -.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))

    def sample_action(self, obs):
        """
            Creates a normal distribution representing
            the current policy which
            if sampled, returns an action on the policy given the observation
        """
        mu = self.logits(obs)
        pi = torch.distributions.Normal(loc=mu, scale=torch.exp(self.log_std))

        return pi

    @classmethod
    def log_p(cls, pi, a):
        """
            The log probability of taken action
            a in policy pi
        """
        return pi.log_prob(a).sum(axis=-1)  # Sum needed for Torch normal distr.


class CategoricalPolicy(Actor):
    """
        Categorical Policy for discrete action spaces
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.Tanh, size=2):
        super(CategoricalPolicy, self).__init__()

        self.logits = mlp(obs_dim, hidden_sizes +
                          [act_dim], activation, size=size)

    def sample_action(self, obs):
        """
            Get new policy
        """
        logits = self.logits(obs)
        pi = torch.distributions.Categorical(logits=logits)

        return pi

    @classmethod
    def log_p(cls, p, a):
        """
            Log probabilities of actions w.r.t pi
        """

        return p.log_prob(a)


class MLPActor(nn.Module):
    """
        Agent actor Net
    """

    def __init__(self, obs_space, act_space, hidden_size=[32, 32], activation=nn.Tanh, size=2):
        super(MLPActor, self).__init__()

        obs_dim = obs_space.shape[0]

        discrete = True if isinstance(act_space, Discrete) else False
        act_dim = act_space.n if discrete else act_space.shape[0]

        if discrete:
            self.pi = CategoricalPolicy(
                obs_dim, act_dim, hidden_size, size=size, activation=activation)
        else:
            self.pi = MLPGaussianPolicy(
                obs_dim, act_dim, hidden_size, activation=activation, size=size)

        self.v = MLPCritic(
            obs_dim, act_dim, hidden_sizes=hidden_size, size=size, activation=activation)

    def step(self, obs):
        """
            Get value function estimate and action sample from pi
        """
        with torch.no_grad():
            pi_new = self.pi.sample_action(obs)
            a = pi_new.sample()

            v = self.v(obs)
            log_p = self.pi.log_p(pi_new, a)

        return a.numpy(), v.numpy(), log_p.numpy()
