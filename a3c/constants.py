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

    actions_arr = np.load('actions_array_useful.npz')
    actions_array = actions_arr['actions_array_useful']
