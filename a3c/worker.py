"""
    A3C agent class for spawning threads
"""

import typing

from threading import Thread

import grid2op
from grid2op.Reward import L2RPNSandBoxScore

import torch
import Torch.Tensor as Tensor
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
        self.clip_ratio = .2

    def run(self):
        """
            Start thread
        """

        global episode, episode_test

        episode = 0

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

                action, log_p = self.get_action(env, state)

                if min(state.rho < .8):
                    action = 0
                else:
                    # If directly from prediction index[0]
                    action = self.get_action(env, state)

                action_vect = constants.actions_array[action:, ]

                act = env.action_space({})
                act.from_vector(action_vect)

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
                    self._log(logs)

                    constants.scores.append(score)
                    episode += 1

                    print('time window: {env.chronics_handler.max_timestep()}')
                    self.update(not done)
                    break

            if not time_step % self.batch_size:
                self.update(not done)
            self._log(logs)

    def _log(self, logs):
        """
            Log episode data
        """

        print('\n', '-' * 10)
        for k, v in logs:
            print(k, v)

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
        act = self.actor.step(state)
        return 0, 0

    @ classmethod
    def process_reward(cls, rew: float) -> float:
        """
            Scales the raw reward
        """

        return 50 - rew / 100

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

    def update(self, eps_terminated: bool = True):
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

        def update_policy():

            self.pi_optim.zero_grad(none=True)

            pi_loss, k_l = self._compute_pi_loss(
                obs_b, act_b, advantages, self.log_p)

            pi_loss.backward()
            self.pi_optim.step()

        def update_v():
            self.v_optim.zero_grad(none=True)
            v_loss = self._compute_v_loss(obs_b, rew_b)

            v_loss.backward()
            self.v_optim.step()
