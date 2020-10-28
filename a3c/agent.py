"""
    Implementation of the a3c agent (Multiple parallel workers)
"""

import os
import typing
import time


import torch
import torch.nn as nn
import torch.optim as optim

import core

from worker import Worker
import constants


class A3C(nn.Module):
    def __init__(self, state_size, action_size):
        self.actor_lr = .001
        self.critic_lr = .005
        self.gamma = .98  # discount factor
        self.lamda = None
        self.n_steps = 10000  # Env training steps

        self.state_size = state_size
        self.action_size = action_size

        cuda = torch.cuda.is_available()
        self.n_workers = torch.cuda.device_count(

        ) if cuda else os.cpu_count()

        self.device = torch.device('cpu' if not cuda else 'cuda')
        torch.backends.cudnn.benchmark = True

        self.actor, self.critic = self.build_nets(state_size, action_size)

        constants.init()
        print('const eps steps: ', constants.EPISODE_STEPS)

    def build_nets(self, state_size: int, action_size: int) -> typing.Tuple[nn.Module, nn.Module]:
        """
            Initializes the actor and critic modules, returning an instance for each
        """
        # Actor and critic share the first layer

        actor = core.CategoricalPolicy(state_size, action_size)
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

    def forward(self, x):
        ...

    def train_workers(self):
        """
            Runs the training loop for A3C agents

            Separate threads are started for the
            number of workers
        """
        args = dict(actor=self.actor,
                    critic=self.critic,
                    gamma=self.gamma,
                    lamda=self.lamda or self.gamma / 1.005,
                    device=self.device)
        workers = [Worker(i, self.action_size, self.state_size, **args)
                   for i in range(self.n_workers)]

        for worker in workers:
            worker.start()

        while len(constants.scores) < self.n_steps:
            time.sleep(400)  # save checkpoint every 400 ms

            print(f'\nCurrent scores: {constants.scores}')

            self.save(constants.episode)
            print('\nCheckpoint saved at episode: {constants.episode}\n')
