from ppo import PPO

import gym
# import grid2op
#from lightsim2grid.LightSimBackend import LightSimBackend


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

    ac_args = {'hidden_size': [64, 64], 'size': 2}
    train_args = {
        'pi_train_n_iters': 80,
        'v_train_n_iters': 80,
        'max_kl': .01,  # TODO remember to take this back to .01
        'max_eps_len': 150,
        'clip_ratio': .2
    }
    agent_args = {
        'n_epochs': 100,
        'env_name': '',  # 'b_10000_plr_.1e-4',
        'steps_per_epoch': 10000
    }

    args = {
        'ac_args': ac_args,
        'pi_lr': 2e-4,
        'v_lr': 1e-3,
        'gamma': .99,
        'lamda': .97,
        **agent_args,
        **train_args
    }

    runner = PPO(env, **args)
    runner.run_training_loop()


if __name__ == '__main__':
    main()
