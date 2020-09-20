import os

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


def main():
    """
        Ppo runner
    """

    env = grid2op.make("l2rpn_case14_sandbox", backend=backend)

    ac_args = {
        'pi': {
            'hidden_sizes': [800, 800, 512, 512],
            'size': 5,  # TODO change size to min 5
            'activation': nn.Tanh
        },
        'v': {
            'hidden_sizes': [800, 800, 512, 512, 256][:-4],
            'size': 6,
            'activation': nn.Tanh
        }
    }
    train_args = {
        'pi_train_n_iters': 80,
        'v_train_n_iters': 80,
        'max_kl': .015,  # TODO remember to take this back to .01
        'max_eps_len': 1000,
        'clip_ratio': .2
    }
    feature_args = {
        # observation attr used in training

        # TODO restore commented out obs attributes
        'obs_attributes': [
            "day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v",
            "load_p", "load_q", "actual_dispatch", "target_dispatch",
            "topo_vect", "time_before_cooldown_line",
            "time_before_cooldown_sub", "timestep_overflow", "line_status"
            # "rho",'month'
        ],

        # Actions agent can do
        'kwargs_converters': {
            'all_actions': None,
            'set_line_status': False,
            'set_topo_vect': True,
            'redispatch': True,
            'change_bus_vect': True
        },

        # Whether to perform action filtering
        # See {AgentClassName}._filter_act for info
        'filter_acts':
        True
    }

    agent_args = {
        'n_epochs': 1000,
        'env_name': '',  # 'b_10000_plr_.1e-4',
        'steps_per_epoch': 10000,
        'save_frequency': 10,
        'training': True,

        # If true use torch.torch.optim.lr_scheduler.ReduceLROnPlateau
        'schedule_pi_lr': True,
        'schedule_v_lr': False
    }

    # Log step count 10 times
    agent_args['log_step_freq'] = agent_args['steps_per_epoch'] / 10

    args = {
        'ac_args': ac_args,
        'pi_lr': 1e-3,
        'v_lr': 1e-3,
        'gamma': .99,
        'lamda': .995,
        'save_path': 'PPO_MODEL.pt',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        **agent_args,
        **train_args,
        **feature_args
    }

    dir_name = os.path.dirname(os.path.abspath('__file__'))
    save_dir = os.path.join(dir_name, '.models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args['save_path'] = os.path.join(save_dir, args.get('save_path'))

    runner = PPOAgent(env, env.observation_space, env.action_space, **args)
    print('Running training loop..')
    runner.train()


if __name__ == '__main__':
    main()
