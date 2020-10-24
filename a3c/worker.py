"""
    A3C agent class for spawning threads
"""

from threading import Thread

import grid2op

from grid2op.Reward import L2RPNSandBoxScore

import constants

import numpy as np


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

        self.worker_idx = index
        self.actor = args['actor']
        self.critic = args['critic']
        self.optimizers = args['optimizers']

        self.gamma = args['gamma']
        self.action_dim = action_dim
        self.state_size = state_size

        self.env_name = env_name
        self.batch_size = batch_size

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

                if min(state.rho < .8):
                    action = 0
                else:
                    action = self.get_action(state, env, state)

                action_vect = constants.actions_array[action:, ]

                act = env.action_space({})
                act.from_vector(action_vect)

                n_state, rew, done, info = env.step(act)

                reward = self.process_reward(reward)

                if done:
                    score += -100  # Penalty for grid failure
                    self.store(state, action, reward)

                    p_d = np.sum(state.prod_p) - np.sum(state.load_p)
                    print(f'Done at episode: {episode}')
                    print(f'Env timestep: {env.time_stamp}')
                    print(f'Power deficiency: {p_d}')
                else:
                    ...

    def store(self, state, action, reward):
        """
            Stores a transition in memory
        """

        self.states.append(state)
        act = np.zeros(self.action_dim)

        act[action] = 1

        self.actions.append(act)
        self.rewards.append(reward)

    def get_action(self, env, state):
        """
            Predicts an action for a given state
        """
        ...
        return 0

    @classmethod
    def process_reward(cls, rew: float) -> float:
        """
            Scales the raw reward
        """

        return 50 - rew / 100
