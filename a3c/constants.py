"""
    Shared constants across parallel agents
"""

import numpy as np


def init():
    """
        Initializes all constants

    """
    global EPISODE_STEPS
    EPISODE_STEPS = 10000

    global actions_array

    act_name = 'actions_array'
    # act_name = 'actions_array_useful'
    actions_arr = np.load(f'{act_name}.npz')
    actions_array = actions_arr[act_name].T

    global scores
    scores = []

    global episode
    episode = 0
