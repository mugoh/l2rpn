from ppo import PPO
import torch.nn as nn

import gym
# import grid2op
# from lightsim2grid.LightSimBackend import LightSimBackend


def main():
    """
        Ppo runner
    """

    #   backend = LightSimBackend()
    #   env = grid2op.make("l2rpn_neurips_2020_track1",
    #                      test=True,
    #                      difficulty="0",
    #                      backend=backend)
    env = gym.make('HalfCheetah-v2')

    ac_args = {
        'hidden_size': [64, 64],
        'size': 2,
        'pi': {
            'hidden_size': [800, 800, 512, 512],
            'size': 4,
            'activation': nn.Tanh
        },
        'v': {
            'hidden_size': [800, 800, 512, 512, 256],
            'size': 5,
            'activation': nn.Tanh
        }
    }
    train_args = {
        'pi_train_n_iters': 80,
        'v_train_n_iters': 80,
        'max_kl': .01,  # TODO remember to take this back to .01
        'max_eps_len': 150,
        'clip_ratio': .2
    }
    feature_args = {
        # observation attr used in training
        'obs_attributes': [
            "day_of_week", "hour_of_day", "minute_of_hour", "prod_p", "prod_v",
            "load_p", "load_q", "actual_dispatch", "target_dispatch",
            "topo_vect", "time_before_cooldown_line",
            "time_before_cooldown_sub", "rho", "timestep_overflow",
            "line_status"
        ],

        # Actions agent can do
        'kwargs_converters': {
            'all_actions': None,
            'set_line_status': False,
            'set_topo_vect': False,
            'redispatch': True,
            'change_bus_vect': True
        },

        # Whether to perform action filtering
        # See {AgentClassName}._filter_act for info
        'filter_acts':
        True
    }

    agent_args = {
        'n_epochs': 100,
        'env_name': '',  # 'b_10000_plr_.1e-4',
        'steps_per_epoch': 10000,
        'save_frequency': 100,
        'training': True,

        # If true use torch.torch.optim.lr_scheduler.ReduceLROnPlateau
        'schedule_pi_lr': False,
        'schedule_v_lr': False
    }

    args = {
        'ac_args': ac_args,
        'pi_lr': 3e-4,
        'v_lr': 1e-3,
        'gamma': .99,
        'lamda': .97,
        **agent_args,
        **train_args,
        **feature_args
    }

    runner = PPO(env, **args)
    runner.run_training_loop()


if __name__ == '__main__':
    main()
