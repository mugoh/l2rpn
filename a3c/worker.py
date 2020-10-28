"""
    A3C agent class for spawning threads
"""

import typing
import time
from threading import Thread
import os

import grid2op
from grid2op.Reward import L2RPNSandBoxScore

import torch
import torch.Tensor as Tensor
from torch.utils.tensorboard import SummaryWriter

import numpy as np


import constants
import core


class Worker(Thread):
    """
        The Agent class.

        Spawns multiple copies of the environment each with its own
        worker. All workers share weights to a common global network

        Parameters
        ----------
        index (int): Worker ID
    """

    def __init__(self, index: int, action_dim, state_size, env_name: str = 'l2rpn_wcci_2020', batch_size: int = 256, ** args):
        self.states = []
        self.rewards = []
        self.actions = []
        self.log_p = []

        self.worker_idx = index
        self.actor = args['actor']
        self.critic = args['critic']
        self.pi_optim, self.v_optim = args['optimizers']

        self.gamma = args['gamma']
        self.lamda = args['lamda']
        self.action_dim = action_dim
        self.state_size = state_size

        self.env_name = env_name
        self.batch_size = batch_size

        self.device = args['device']
        self.v_critierion = torch.nn.MSELoss()
        self.clip_ratio = .2  # PPO advantage clip ratio
        self.epsilon = .01  # exploration tradeoff

        self._setup_logger(env_name)

    def _setup_logger(self, env_name: str):
        run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.path_time = time.strftime('%Y-%m-%d')  # For model path
        path = os.path.join('data',
                            env_name  # + args.get('env_name', '')
                            + '_' + run_t)

        self.logger = SummaryWriter(log_dir=path)

    def run(self):
        """
            Start thread
        """

        global episode, episode_test

        episode = 0
        eps_scores = []

        print(f'Starting agent: {self.worker_idx}\n')

        env = grid2op.make(
            self.env_name, reward_class=L2RPNSandBoxScore, difficulty='competition')

        while episode < constants.EPISODE_STEPS:
            env.set_id(episode)
            state = env.reset()

            time_step_end = env.chronics_handler.max_timestep() - 2

            print('time step end: ', time_step_end)

            time_hour = 0
            score = 0

            time_step = 0
            non_zero_actions = 0

            while True:

                # If directly from prediction index[0]
                action, log_p = self.get_action(env, state)

                if min(state.rho < .8):
                    action = 0

                action_vect = constants.actions_array[action:, ]

                act = env.action_space({})
                act.from_vect(action_vect)

                n_state, rew, done, _ = env.step(act)

                reward = self.process_reward(rew)

                if done:
                    score += -100  # Penalty for grid failure
                    self.store(self._convert_obs(state), action, log_p, -100)

                    print(f'Done at episode: {episode}')
                    print(f'Env timestep: {env.time_stamp}')

                    state = np.zeros([1, self.state_size])  # reset
                else:
                    state = n_state
                    # elapsed time in hours
                    time_hour = state.hour_of_day + state.day * 24

                    # Penalize lines close to the current limit
                    # Current limit is reached when a line is overloaded
                    # e.g a line disconnects leaving unbalanced power in
                    # the others
                    over_lm_current = 50 * \
                        np.sum((state.rho - 1)[state.rho > 1])

                    score += (reward - over_lm_current)

                    score = -10 if score < -10 else score
                    self.store(self._convert_obs(state), action, log_p, score)

                eps_scores.append(score)

                p_d = np.sum(state.prod_p) - np.sum(state.load_p)
                print(f'Power deficiency: {p_d}')

                time_step += 1
                non_zero_actions += 0 if not action else 1

                terminal = done or time_step >= time_step_end

                logs = {
                    'episode': episode,
                    'average score': score/time_step,
                    'final action': action,
                    'end step': time_step,
                    'no. non-zero-actions': non_zero_actions,
                    'time-hr': time_hour
                }

                if terminal:
                    print(
                        f'\nStopped Thread: {self.worker_idx}: done: {done}\n')
                    self._log(logs, eps_scores, episode)

                    constants.scores.append(score)
                    episode += 1

                    print('time window: {env.chronics_handler.max_timestep()}')
                    self.update(episode, not done)
                    eps_scores = []
                    break

            if not time_step % self.batch_size:
                self.update(episode, not done)
            self._log(logs, eps_scores, episode)

    def _log(self, logs: typing.Dict, eps_scores: typing.Iterable, eps: int):
        """
            Log episode data
        """

        print('\n', '-' * 10)
        for key, val in logs.items():
            print(key, val)

        self.logger.add_scalar('Reward/Mean', np.mean(constants.scores), eps)
        self.logger.add_scalar('Reward/Max', np.max(constants.scores), eps)
        self.logger.add_scalar('Reward/Min', np.min(constants.scores), eps)
        self.logger.add_scalar('EPS_Score/Average', np.mean(eps_scores), eps)

    def store(self, state, action, log_p, reward):
        """
            Stores a transition in memory
        """

        self.states.append(state)
        act = np.zeros(self.action_dim)

        act[action] = 1

        self.actions.append(act)
        self.rewards.append(reward)
        self.log_p.append(log_p)

    def _convert_obs(self, state: np.array) -> np.array:
        """
            Alters the features of the vector observation
            to give a state with usable properties
        """
        obs = state.to_vect()
        numerical_obs = obs[6:713]

        numerical_obs[649:706] = numerical_obs[649:706] * 100

        topological_vect = state.topo_vect
        usable_obs = np.hstack(
            (numerical_obs, topological_vect, topological_vect - 1, state.line_status))

        return usable_obs

    def get_action(self, env, state) -> typing.Iterable:
        """
            Predicts an action for a given state
        """
        state_input = torch.from_numpy(
            self._convert_obs(state)).to(self.device)

        if np.random.uniform() < self.epsilon:
            self.epsilon = np.max([.01, self.epsilon * .995])

            action = np.random.randint(self.action_dim, size=1)
            _, log_p = self.actor.step(
                state_input,
                act=torch.from_numpy(action)
                .type(torch.float32)
                .to(self.device))

            return action, log_p

        act_idx, log_p, action_probs = self.actor.step(
            state_input, act_probs=True)

        act_idx = act_idx[0]

        action_cls = env.action_space({})
        action_cls.from_vect(constants.actions_array[act_idx, :])

        # Simulate
        obs_, rew_, done_, _ = state.simulate(action_cls)

        if done_ or np.sum((obs_.rho - 1)[obs_.rho > 1.02]) > 0:
            rew_ = self.process_reward(rew_)
            rew_ = self.estimate_rew_update(obs_, rew_, done_)

            additional_act = 1007
            np.argsort(action_probs)[-1: -additional_act - 1:, -1]

            action_cls = np.zeros(additional_act)
            siml_reward = np.zeros(additional_act)

            for i in range(additional_act):
                action_cls[i] = env.action_space({})
                action_cls[i].from_vect(
                    constants.actions_array[action_probs[i], :])

                obs, siml_reward, done, _ = state.simulate(action_cls[i])
                siml_reward[i] = self.process_reward(siml_reward[i])
                siml_reward[i] = self.estimate_rew_update(
                    obs, siml_reward[i], done)

                if not done and sum((obs.rho - 1)[obs.rho > 1]) == 0:
                    if i > 20:
                        print('Simulated action not done')
                        _logs = dict(
                            current_act=i,
                            max_rew=np.max(siml_reward),
                            power_deff=sum((obs.rho - 1)[obs.rho > 1.02])
                        )

                        for key, val in _logs.items():
                            print(key, ' -- ', val)

    @ classmethod
    def process_reward(cls, rew: float) -> float:
        """
            Scales the raw reward
        """

        return 50 - rew / 100

    def estimate_rew_update(self, obs: typing.List, rew: typing.List, done: bool) -> float:
        """
            Estimates the reward update from penalty on overloaded
            lines on the current state
        """
        rew = rew - 5 * sum((obs.rho - 1)[obs.rho > 1]) if not done else -100

        return rew

    def _compute_v_loss(self, obs_b, rew_b):
        """
            Finds value function loss
        """

        pred = self.critic(obs_b)

        v_loss = self.v_critierion(pred, rew_b)

        return v_loss

    def _compute_pi_loss(self, obs_b: Tensor, act_b: Tensor, adv_b: Tensor, old_logp: Tensor) -> typing.Tuple:
        """
            Computes loss for the policy
        """
        _, log_p = self.actor(obs_b, act_b)

        clip = self.clip_ratio

        ratio = torch.exp(log_p - old_logp)

        min_adv = torch.where(adv_b >= 0, (1+clip) * adv_b, (1-clip) * adv_b)

        loss = -torch.min(min_adv, ratio).mean()
        k_l = (old_logp - log_p).mean().item()

        return loss, k_l

    def update(self, epoch: int, eps_terminated: bool = True):
        """
            Trains the network at the end of each
            episode
        """

        if eps_terminated:
            # estimate value of end state
            # when episode is terminated at max_eps
            final_v = self.critic.predict_v(torch.from_numpy(
                self.states[-1])
                .dtype(torch.float32)
                .view(1, -1)
                .to(self.device)
            )
        else:
            final_v = 0

        disc_rewards = core.disc_cumsum(self.rewards, self.gamma)[:-1]

        values = self.critic.predict_v(torch.from_numpy(
            self.states)
            .type(torch.float32)
            .to(self.device)
        )

        # Estimate advantages using GAE
        deltas = self.rewards[:-1] + self.gamma * values[1:] - values[:-1]
        advantages = core.disc_cumsum(deltas, self.gamma * self.lamda)

        obs_b, rew_b, act_b = self.states, self.rewards, self.actions

        obs_b, rew_b, act_b = self._get_tensors(obs_b, rew_b, act_b)

        def update_policy():

            self.pi_optim.zero_grad(none=True)

            pi_loss, k_l = self._compute_pi_loss(
                obs_b, act_b, advantages, self.log_p)

            pi_loss.backward()
            self.pi_optim.step()

            self.logger.add_scalar('loss/pi', pi_loss, epoch)
            self.logger.add_scalar('KL', k_l, epoch)

        def update_v():
            self.v_optim.zero_grad(none=True)
            v_loss = self._compute_v_loss(obs_b, rew_b)

            v_loss.backward()
            self.v_optim.step()

            self.logger.add_scalar('loss/V', v_loss, epoch)

        update_policy()
        update_v()

        self.states, self.actions, self.rewards, self.log_p = [], [], [], []

    def _get_tensors(self, *args: typing.Iterable) -> typing.List:
        return [torch.as_tensor(arg, dtype=torch.float32, device=self.device) for arg in args]
