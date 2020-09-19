import numpy as np
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import gym
import time
import os

import core
from core import dcum2 as discounted_cumsum


class ReplayBuffer:
    """
        Transitions buffer
        Stores transitions for a single episode
    """
    def __init__(self, act_dim, obs_dim, size=4000, gamma=.98, lamda=.95):
        self.size = size
        self.gamma = gamma
        self.lamda = lamda

        self.rewards = np.zeros([size], dtype=np.float32)
        self.actions = np.zeros([size, act_dim], dtype=np.float32)
        self.states = np.zeros([size, obs_dim], dtype=np.float32)

        self.log_prob = np.zeros([size], dtype=np.float32)
        self.adv = np.zeros([size], dtype=np.float32)
        self.vals = np.zeros([size], dtype=np.float32)

        self.ptr, self.eps_end_ptr = 0, 0

    def store(self, act, states, values, rew, log_p):
        """
            Store transitions
        """
        idx = self.ptr % self.size

        self.rewards[idx] = rew
        self.actions[idx] = act
        self.states[idx] = states
        self.vals[idx] = values
        self.log_prob[idx] = log_p

        self.ptr += 1

    def get(self):
        """
            Returns episode transitions
        """
        assert self.ptr >= self.size

        self.ptr = 0
        self.eps_end_ptr = 0
        return torch.from_numpy(self.actions), torch.from_numpy(self.rewards), \
            torch.from_numpy(self.states), torch.from_numpy(
            self.adv), torch.from_numpy(self.log_prob)

    def end_eps(self, value=0):
        """
            Calculates the adv once the agent
            encounters an end state

            value: value of that state -> zero if the agent
            died or the value function if the episode was terminated
        """
        idx = slice(self.eps_end_ptr, self.ptr)

        rew = np.append(self.rewards[idx], value)
        vals = np.append(self.vals[idx], value)

        # GAE
        deltas = rew[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv[idx] = discounted_cumsum(deltas, self.gamma * self.lamda)

        # Reward to go
        self.rewards[idx] = discounted_cumsum(rew, self.gamma)[:-1]

        self.eps_end_ptr = self.ptr


class PPO:
    def __init__(self, env, actor_class=core.MLPActor, **args):
        """
        actor_args: hidden_size(list), size(int)-network size, pi_lr, v_lr
        max_lr: Max kl divergence between new and old polices (0.01 - 0.05)
                Triggers early stopping for pi training
        """

        obs_space = env.observation_space
        act_space = env.action_space

        act_dim = act_space.shape[0] if not isinstance(
            act_space, gym.spaces.Discrete) else act_space.n
        obs_dim = obs_space.shape[0]
        self.args = args
        self.env = env

        self.actor = actor_class(obs_space=obs_space,
                                 act_space=act_space,
                                 **args['ac_args'])
        params = [
            core.count(module) for module in (self.actor.pi, self.actor.v)
        ]
        print(f'\nParameters\npi: {params[0]}  v: { params[1] }')

        self.memory = ReplayBuffer(act_dim,
                                   obs_dim,
                                   args['steps_per_epoch'],
                                   lamda=args['lamda'],
                                   gamma=args['gamma'])

        self.pi_optimizer = optim.Adam(self.actor.pi.parameters(),
                                       args['pi_lr'])
        self.v_optimizer = optim.Adam(self.actor.v.parameters(), args['v_lr'])

        # Hold epoch losses for logging
        self.pi_losses, self.v_losses, self.delta_v_logs, self.delta_pi_logs = [], [], [], []
        self.pi_kl = []  # kls for logging
        self.v_logs = []
        self.first_run_ret = None

        run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
        path = os.path.join(
            'data',
            env.unwrapped.spec.id + args.get('env_name', '') + '_' + run_t)

        self.logger = SummaryWriter(log_dir=path)

    def _compute_pi_loss(self, log_p_old, adv_b, act_b, obs_b):
        """
            Pi loss
        """
        clip_ratio = self.args['clip_ratio']

        # returns new_pi_normal_distribution, logp_act
        _, log_p_ = self.actor.pi(obs_b, act_b)
        log_p_ = log_p_.type(torch.float32)  # From torch.float64

        pi_ratio = torch.exp(log_p_ - log_p_old)
        min_adv = torch.where(adv_b >= 0, (1 + clip_ratio) * adv_b,
                              (1 - clip_ratio) * adv_b)

        pi_loss = -torch.mean(torch.min(pi_ratio * adv_b, min_adv))

        return pi_loss, (log_p_old - log_p_).mean().item()  # kl

    def _compute_v_loss(self, data):
        """
            Value function loss
        """
        obs_b, rew_b = data['obs_b'], data['rew_b']

        v_pred = self.actor.v(obs_b)
        v_loss = ((v_pred - rew_b)**2).mean()

        return v_loss

    def _update(self, epoch):
        """
            Update the policy and value function from loss
        """
        data = self.memory.get()
        act_b, rew_b, obs_b, adv_b, log_p_old = data
        train_args = self.args

        # loss before update
        pi_loss_old, kl = self._compute_pi_loss(log_p_old=log_p_old,
                                                obs_b=obs_b,
                                                adv_b=adv_b,
                                                act_b=act_b)

        v_loss_old = self._compute_v_loss({
            'obs_b': obs_b,
            'rew_b': rew_b
        }).item()

        for i in range(train_args['pi_train_n_iters']):
            self.pi_optimizer.zero_grad()
            pi_loss, kl = self._compute_pi_loss(log_p_old=log_p_old,
                                                obs_b=obs_b,
                                                adv_b=adv_b,
                                                act_b=act_b)

            if kl > 1.5 * train_args['max_kl']:  # Early stop for high Kl
                print('Max kl reached: ', kl, '  iter: ', i)
                break

            pi_loss.backward()
            self.pi_optimizer.step()

        self.logger.add_scalar('PiStopIter', i, epoch)
        pi_loss = pi_loss.item()

        for i in range(train_args['v_train_n_iters']):
            self.v_optimizer.zero_grad()
            v_loss = self._compute_v_loss({'obs_b': obs_b, 'rew_b': rew_b})

            v_loss.backward()
            self.v_optimizer.step()

        v_loss = v_loss.item()

        self.pi_losses.append(pi_loss)
        self.pi_kl.append(kl)
        self.v_losses.append(v_loss)

        delta_v_loss = v_loss_old - v_loss
        delta_pi_loss = pi_loss_old.item() - pi_loss

        self.delta_v_logs.append(delta_v_loss)
        self.delta_pi_logs.append(delta_pi_loss)

        self.logger.add_scalar('loss/pi', pi_loss, epoch)
        self.logger.add_scalar('loss/v', v_loss, epoch)

        self.logger.add_scalar('loss/Delta-Pi', delta_pi_loss, epoch)
        self.logger.add_scalar('loss/Delta-V', delta_v_loss, epoch)
        self.logger.add_scalar('Kl', kl, epoch)

    def run_training_loop(self):
        start_time = time.time()
        obs = self.env.reset()
        eps_len, eps_ret = 0, 0

        n_epochs = self.args['n_epochs']
        steps_per_epoch = self.args['steps_per_epoch']
        max_eps_len = self.args['max_eps_len']

        for t in range(n_epochs):
            eps_len_logs, eps_ret_log = [], []
            for step in range(steps_per_epoch):
                a, v, log_p = self.actor.step(
                    torch.from_numpy(obs).type(torch.float32))

                # log v
                self.v_logs.append(v)
                obs_n, rew, done, _ = self.env.step(a)

                eps_len += 1
                eps_ret += rew

                self.memory.store(a, obs, values=v, log_p=log_p, rew=eps_ret)

                obs = obs_n

                terminal = done or eps_len == max_eps_len

                if terminal or step == steps_per_epoch - 1:
                    # terminated by max episode steps
                    if not done:
                        last_v = self.actor.step(
                            torch.from_numpy(obs).type(torch.float32))[1]
                    else:  # Agent terminated episode
                        last_v = 0

                    if terminal:
                        # only log these for terminals
                        eps_len_logs += [eps_len]
                        eps_ret_log += [eps_ret]

                    self.memory.end_eps(value=last_v)
                    obs = self.env.reset()
                    eps_len, eps_ret = 0, 0

            self._update(t + 1)
            l_t = t + 1  # log_time, start at 1

            # Print info for each epoch: loss_pi, loss_v, kl
            # time, v at traj collection, eps_len, epoch_no,
            # eps_ret: min, max, av
            AverageEpisodeLen = np.mean(eps_len_logs)

            self.logger.add_scalar('AvEpsLen', AverageEpisodeLen, l_t)
            # MaxEpisodeLen = np.max(eps_len_logs)
            # MinEpsiodeLen = np.min(eps_len_logs)
            AverageEpsReturn = np.mean(eps_ret_log)
            MaxEpsReturn = np.max(eps_ret_log)
            MinEpsReturn = np.min(eps_ret_log)

            self.logger.add_scalar('EpsReturn/Max', MaxEpsReturn, l_t)
            self.logger.add_scalar('EpsReturn/Min', MinEpsReturn, l_t)
            self.logger.add_scalar('EpsReturn/Average', AverageEpsReturn, l_t)

            # Retrieved by index, not time step ( no +1 )
            Pi_Loss = self.pi_losses[t]
            V_loss = self.v_losses[t]
            Kl = self.pi_kl[t]
            delta_v_loss = self.delta_v_logs[t]
            delta_pi_loss = self.delta_pi_logs[t]

            if t == 0:
                self.first_run_ret = AverageEpsReturn

            logs = {
                'EpsReturn/Average': AverageEpsReturn,
                'EpsReturn/Max': MaxEpsReturn,
                'EpsReturn/Min': MinEpsReturn,
                'AverageEpsLen': AverageEpisodeLen,
                'KL': Kl,
                'Pi_Loss': Pi_Loss,
                'V_loss': V_loss,
                'FirstEpochAvReturn': self.first_run_ret,
                'Delta-V': delta_v_loss,
                'Delta-Pi': delta_pi_loss,
                'RunTime': time.time() - start_time
            }

            print('\n', t + 1)
            print('-' * 35)
            for k, v in logs.items():
                print(k, v)
            print('\n\n\n')


def main():
    """
        Ppo runner
    """

    env = gym.make('HalfCheetah-v2')

    ac_args = {'hidden_size': [64, 64], 'size': 2}
    train_args = {
        'pi_train_n_iters': 80,
        'v_train_n_iters': 80,
        'max_kl': .01,
        'max_eps_len': 150
    }
    agent_args = {
        'n_epochs': 100,
        'env_name': '',  # 'b_10000_plr_.1e-4',
        'steps_per_epoch': 10000
    }

    args = {
        'ac_args': ac_args,
        'pi_lr': 1e-4,
        'v_lr': 1e-3,
        'gamma': .99,
        'lamda': .97,
        **agent_args,
        **train_args
    }


if __name__ == '__main__':
    main()
