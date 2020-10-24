"""
    Implementation of the a3c agent (Multiple parallel workers)
"""

import os
import typing

import torch
import torch.nn as nn
import torch.optim as optim

import core


class A3C(nn.Module):
    def __init__(self, state_size, action_size):
        self.actor_lr = .001
        self.critic_lr = .005
        self.gamma = .98  # discount factor

        cuda = torch.cuda.is_available()
        self.n_workers = torch.cuda.device_count(

        ) if cuda else os.cpu_count()

        self.device = 'cpu' if not cuda else 'cuda'

        self.actor, self.critic = self.build_nets(state_size, action_size)

    def build_nets(self, state_size: int, action_size: int) -> typing.Tuple[nn.Module, nn.Module]:
        """
            Initializes the actor and critic modules, returning an instance for each
        """
        # Actor and critic share the first layer

        actor = core.Actor(state_size, action_size)
        critic = core.Critic(shared_layer=actor.fc1,
                             state_size=state_size, act_dim=action_size)

        return actor.to(self.device), critic.to(self.device)

    def configure_optimizers(self):

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.critic_lr)

    def load_checkpoint(self, path: str = '', train: bool = True) -> int:
        """
            Retrives saved model checkpoint

            Returns the saved training epoch
        """

        if not path:
            dir_ = os.path.dirname(os.path.realpath('__file__'))
            path = os.path.join(dir_, '.model.pt')

        try:
            ckpt = torch.load(path)
        except FileNotFoundError:
            return 0
        else:
            print('Loaded model at epoch: ', end='')

        self.load_state_dict(ckpt['model_state_dict'])
        self.actor_optimizer.load_state_dict(ckpt['ac_optim_dict'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optim_dict'])
        epoch = ckpt['epoch']

        print(epoch)

        if not train:
            self.eval()
        else:
            self.train()

        return epoch

    def save(self, epoch: int, path: str = '.model.pt'):
        """
            Saves the current model checkpoint

            Parameters:
            epoch (int): Training epoch at time of saving
        """
        state_dict = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'ac_optim_dict': self.actor_optimizer.state_dict(),
            'critic_optim_dict': self.critic_optimizer.state_dict()
        }

        torch.save(state_dict, path)
