import os

import json
import time

import torch.nn as nn
import torch

from ppo import PPOAgent

import grid2op

try:
    from lightsim2grid.LightSimBackend import LightSimBackend as backend
except ModuleNotFoundError:
    from grid2op.Backend import PandaPowerBackend as backend
finally:
    backend = backend()

from grid2op.Reward import L2RPNReward


class RewardPenalizeIllegal(L2RPNReward):
    """
        Only penalize for taking illegal actions
    """

    def __init__(self, **args):

        super(RewardPenalizeIllegal, self).__init__(**args)

    def initialize(self, env):
        self.reward_min = 0.
        self.reward_max = 1.

    def __call__(self, action, env, has_error, is_done, is_illegal,
                 is_ambiguous):
        if has_error or is_illegal or is_ambiguous:
            rew = -1000
        else:
            rew = super().__call__(action, env, has_error, is_done, is_illegal,
                                   is_ambiguous)
            rew = rew / env.n_line

        return rew


def main():
    """
        Ppo runner
    """

    env = grid2op.make("l2rpn_neurips_2020_track1_large",
                       backend=backend,
                       reward_class=RewardPenalizeIllegal)
    ac_args = {
        # obs: ~170-190, act_dim: ~200
        'pi': {
            'hidden_sizes': [1024, 1024],
            'size': 2,  # TODO change size to min 5
            'activation': nn.Tanh
        },
        'v': {
            'hidden_sizes': [1024, 1024],
            'size': 2,
            'activation': nn.Tanh
        }
    }
    train_args = {
        'pi_train_n_iters': 580,
        'v_train_n_iters': 580,
        'max_eps_len': 150,
        'clip_ratio': .2
    }

    # Anneal target kl by (max - min)/fin_epoch
    kl_args = {
        'max_kl_start': 10.,
        'min_kl_stop': 5,
        'kl_fin_epoch': 100,

        # If false, use target_kl throughout
        'anneal_kl': True,
        'target_kl': .1,  # TODO remember to take this back to .01
    }

    feature_args = {
        # observation attr used in training

        # TODO restore commented out obs attributes
        'obs_attributes': [
            "day_of_week",
            "hour_of_day",
            "minute_of_hour",
            "prod_p",
            # "prod_v",
            "load_p",
            "load_q",
            # "actual_dispatch",
            "target_dispatch",
            "topo_vect",
            "time_before_cooldown_line",
            "time_before_cooldown_sub",
            "timestep_overflow",
            "line_status",
            "rho"
            # 'month'
        ],

        # Actions agent can do
        'kwargs_converters': {
            'all_actions': None,
            'set_line_status': True,
            'set_topo_vect': True,
            'redispatch': True,
            'change_bus_vect': True
        },

        # Whether to perform action filtering
        # See {AgentClassName}._filter_act for info
        'filter_acts':
        True,

        # Filter obs. If not, use entire obs space
        'filter_obs':
        True
    }

    agent_args = {
        'n_epochs': 5000,
        'env_name': '',  # 'b_10000_plr_.1e-4',
        'steps_per_epoch': 250,
        'save_frequency': 5,
        'training': True,

        # If true use torch.torch.optim.lr_scheduler.ReduceLROnPlateau
        'schedule_pi_lr': True,
        'schedule_v_lr': False,

        # Use if schedule_pi_lr is True
        'max_pi_epoch': 100,  # Final epoch to anneal pi lr
        'min_pi_lr': 1e-7  # Final pi lr
    }

    # Log step count 10 times
    agent_args['log_step_freq'] = agent_args['steps_per_epoch'] // 3

    args = {
        'ac_args': ac_args,
        'pi_lr': 3e-5,
        'v_lr': 1e-5,
        'gamma': .98,
        'lamda': .995,
        'save_path': 'PPO_MODEL.pt',  # CUDA runs out of memory
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        **agent_args,
        **train_args,
        **feature_args,
        **kl_args
    }

    dir_name = os.path.dirname(os.path.abspath('__file__'))
    save_dir = os.path.join(dir_name, 'models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args['save_path'] = os.path.join(save_dir, args.get('save_path'))

    # Save params for use in loading trained model dict
    # Activation functions, which are objects are ignored

    t_stmp = time.strftime('%Y-%m-%d-%H-%M-%S')

    with open(f'params_file_{t_stmp}', 'w') as f:
        json.dump(args,
                  f,
                  indent=1,
                  skipkeys=True,
                  default=lambda o: f'converted -> {str(o)}')

    runner = PPOAgent(env, env.observation_space, env.action_space, **args)
    print('Running training loop..')
    runner.train()


if __name__ == '__main__':
    main()
