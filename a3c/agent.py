"""
    Implementation of the a3c agent (Multiple parallel workers)
"""

import os

import torch
import torch.nn as nn


class A3C(nn.Module):
    def __init__(self, state_size, action_size):
        self.actor_lr = .001
        self.critic_lr = .005
        self.gamma = .98  # discount factor

        cuda = torch.cuda.is_available()
        self.n_workers = torch.cuda.device_count(

        ) if cuda else os.cpu_count()
