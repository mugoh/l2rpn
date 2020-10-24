import numpy as np
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast

import time
import os

import core
from core import dcum2 as discounted_cumsum

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct


class ReplayBuffer:
    """
        Transitions buffer
        Stores transitions for a single episode
    """

    def __init__(self,
                 act_dim,
                 obs_dim,
                 size=4000,
                 gamma=.98,
                 lamda=.95,
                 device=None):
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
        self.device = device

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
        return torch.from_numpy(self.actions).to(self.device), torch.from_numpy(self.rewards).to(self.device), \
            torch.from_numpy(self.states).to(self.device), torch.from_numpy(
            self.adv).to(self.device), torch.from_numpy(self.log_prob).to(self.device)

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
        self.adv = (self.adv - self.adv.mean()) / self.adv.std()

        # Reward to go
        self.rewards[idx] = discounted_cumsum(rew, self.gamma)[:-1]

        self.eps_end_ptr = self.ptr


class PPOAgent(AgentWithConverter):
    def __init__(self,
                 env,
                 observation_space,
                 action_space,
                 actor_class=core.MLPActor,
                 **args):
        super(PPOAgent, self).__init__(action_space,
                                       action_space_converter=IdToAct,
                                       **args['kwargs_converters'])
        """
        actor_args: hidden_size(list), size(int)-network size, pi_lr, v_lr
        max_lr: Max kl divergence between new and old polices (0.01 - 0.05)
                Triggers early stopping for pi training
        """

        self.args = args
        self.env = env
        self.device = args['device']

        if args['filter_acts']:
            self.filter_acts = True
            print('Filtering actions..')
            self.action_space.filter_action(self._filter_act)
            print('Done')
        else:
            self.filter_acts = False

        act_dim = self.get_action_size(self.action_space)

        if self.args['filter_obs']:
            print('filtering observations..')
            # For obs extraction
            self._tmp_obs, self._indx_obs = None, None
            obs_dim = self._get_obs_size(observation_space)

            self.extract_obs(observation_space)
            print('done\n')

            self.filter_obs = True
        else:
            obs_dim = observation_space.size()
            self.filter_obs = False

        print('dims', 'obs: ', obs_dim, '  act: ', act_dim)

        self.actor = actor_class(obs_dim,
                                 act_dim,
                                 discrete=True,
                                 device=self.device,
                                 **args['ac_args']).to(self.device)
        params = [
            core.count(module) for module in (self.actor.pi, self.actor.v)
        ]
        print(f'\nParameters\npi: {params[0]}  v: { params[1] }')

        self.memory = ReplayBuffer(act_dim,
                                   obs_dim,
                                   args['steps_per_epoch'],
                                   lamda=args['lamda'],
                                   gamma=args['gamma'],
                                   device=self.device)

        self.training = self.args['training']

        self.max_kl = self.args['max_kl_start']
        self.min_kl = self.args['min_kl_stop']

        self.pi_optimizer = optim.Adam(self.actor.pi.parameters(),
                                       args['pi_lr'])

        if self.args['schedule_pi_lr']:
            self.pi_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.pi_optimizer,
                patience=5,
                verbose=True
                # min_lr=self.args['min_pi_lr']
                # T_max=self.args['max_pi_epoch'], eta_min=self.args['min_pi_lr']
            )
        self.v_optimizer = optim.Adam(self.actor.v.parameters(), args['v_lr'])

        # Hold epoch losses for logging
        self.pi_losses, self.v_losses, self.delta_v_logs, self.delta_pi_logs = [], [], [], []
        self.pi_kl = []  # kls for logging
        self.v_logs = []
        self.first_run_ret = None

        run_t = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.path_time = time.strftime('%Y-%m-%d')  # For model path
        path = os.path.join('data',
                            env.name + args.get('env_name', '') + '_' + run_t)

        self.logger = SummaryWriter(log_dir=path)
        self.pi_scaler = torch.cuda.amp.GradScaler()
        self.v_scaler = torch.cuda.amp.GradScaler()

        print('\n..Init done')

    def _get_obs_size(self, obs_space):
        """
            Get the dimension of an observation
            given the extracted observation attributes
        """
        size = 0

        for attr_name in self.args['obs_attributes']:
            start, end, _ = obs_space.get_indx_extract(attr_name)
            size += (end - start)

        return size

    def get_action_size(self, act_space):
        """
            Gives the size of the action space
            after the extracted action attributes
        """
        convertor = IdToAct(act_space)
        convertor.init_converter(**self.args['kwargs_converters'])

        if self.filter_acts:
            convertor.filter_action(self._filter_act)

        return convertor.n

    def _compute_pi_loss(self, log_p_old, adv_b, act_b, obs_b):
        """
            Pi loss
        """
        clip_ratio = self.args['clip_ratio']

        with autocast():
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

        with autocast():
            v_pred = self.actor.v(obs_b)
            v_loss = ((v_pred - rew_b)**2).mean()

        return v_loss

    def get_kl(self, itr):
        """
            Return KL target based on current epoch
        """
        T_epoch = self.args['kl_fin_epoch']

        if itr > T_epoch:
            return self.min_kl

        rate = (self.max_kl - self.min_kl) / T_epoch

        return self.max_kl - (rate * itr)

    def _filter_act(self, action):
        """
            Wrapper to Filter the action space
            Passed to self.filter_action
        """
        max_elem = 2

        act_dict = action.impact_on_objects()
        elem = 0
        elem += act_dict["force_line"]["reconnections"]["count"]
        elem += act_dict["force_line"]["disconnections"]["count"]
        elem += act_dict["switch_line"]["count"]
        elem += len(act_dict["topology"]["bus_switch"])
        elem += len(act_dict["topology"]["assigned_bus"])
        elem += len(act_dict["topology"]["disconnect_bus"])
        elem += len(act_dict["redispatch"]["generators"])

        if elem <= max_elem:
            return True
        return False

    def extract_obs(self, obs_space):
        """
            Initializes the observation by extracting the
            listed attribute names selected to represent the
            observation.
        """

        tmp = np.zeros(0, dtype=np.uint)  # TODO platform independant
        for obs_attr_name in self.args['obs_attributes']:
            beg_, end_, _ = obs_space.get_indx_extract(obs_attr_name)
            tmp = np.concatenate((tmp, np.arange(beg_, end_, dtype=np.uint)))
        self._indx_obs = tmp
        self._tmp_obs = np.zeros((1, tmp.shape[0]), dtype=np.float32)

    def convert_obs(self, observation):
        """
            Overrides super:

            Converts an observation into a vector then
            selects the attribues identified to represent
            the observation
        """

        obs_vec = observation.to_vect()
        if self.filter_obs:
            self._tmp_obs[:] = obs_vec[self._indx_obs]
            return self._tmp_obs

        return obs_vec

    def my_act(self, transformed_obs, reward=None, done=False):
        """
            Used by the agent to decide on action to take

            Returns an `encoded_action` which is reconverted
          by the inherited `self.convert_act` into a valid
            action that can be taken in the env


        """

        act = self.predict_action(transformed_obs)

        return act

    def train(self):
        """
            Trains actor
        """
        self.run_training_loop()

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

        kl_target = self.get_kl(
            epoch) if self.args['anneal_kl'] else self.args['target_kl']

        for i in range(train_args['pi_train_n_iters']):
            self.pi_optimizer.zero_grad()
            pi_loss, kl = self._compute_pi_loss(log_p_old=log_p_old,
                                                obs_b=obs_b,
                                                adv_b=adv_b,
                                                act_b=act_b)

            # Early stop for high Kl
            if kl > kl_target:
                print('Max kl reached: ', kl, '[target: ', kl_target,
                      '] iter: ', i)
                break

            self.pi_scaler.scale(pi_loss).backward()
            self.pi_scaler.step(self.pi_optimizer)

            self.pi_scaler.update()

        if self.args['schedule_pi_lr']:
            self.pi_scheduler.step(pi_loss)

        self.logger.add_scalar('PiStopIter', i, epoch)
        pi_loss = pi_loss.item()

        for i in range(train_args['v_train_n_iters']):
            self.v_optimizer.zero_grad()
            v_loss = self._compute_v_loss({'obs_b': obs_b, 'rew_b': rew_b})

            self.v_scaler.scale(v_loss).backward()
            self.v_scaler.step(self.v_optimizer)

            self.v_scaler.update()

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

    def predict_action(self, obs):
        """
            Selects an action given an observation
        """

        return self.actor.step(torch.from_numpy(obs).to(self.device),
                               act_only=True)

    def load(self, path='PPO_MODEL.pt'):
        """
            Loads trained actor network
        """
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Loaded model from: {path}')

        if not self.training:
            self.actor.eval()  # sets self.train(False)

    def save(self, path='PPO_MODEL.pt'):
        """
            Saves trained actor net parameters
        """
        path = self.args['save_path']

        name, ext = path.rsplit('.', 1)
        path_name = f'{ name }-{self.path_time}.{ext}'

        torch.save(self.actor.state_dict(), path_name)
        print(f'Saved model at -> {path_name}')

    def run_training_loop(self):
        start_time = time.time()
        obs = self.convert_obs(self.env.reset())
        eps_len, eps_ret = 0, 0

        n_epochs = self.args['n_epochs']
        steps_per_epoch = self.args['steps_per_epoch']
        max_eps_len = self.args['max_eps_len']

        err_act_msg = [
            'is_illegal', 'is_ambiguous', 'is_dispatching_illegal',
            'is_illegal_reco'
        ]
        log_steps = self.args['log_step_freq']

        for t in range(n_epochs):
            eps_len_logs, eps_ret_log = [], []
            for step in range(steps_per_epoch):

                # Taking really long
                if log_steps and not step % log_steps:
                    print(f'epoch: {t}, step: {step}')

                a, v, log_p = self.actor.step(
                    torch.from_numpy(obs).type(torch.float32).to(self.device))
                act = a

                # log v
                self.v_logs.append(v)
                obs_n, rew, done, info = self.env.step(self.convert_act(a[0]))

                obs_n = self.convert_obs(obs_n)

                # Invalid action
                _ = [
                    print(a, err_msg) for err_msg in err_act_msg
                    if info[err_msg]
                ]

                eps_len += 1
                eps_ret += rew

                self.memory.store(a, obs, values=v, log_p=log_p, rew=rew)

                obs = obs_n

                terminal = done or eps_len == max_eps_len

                if terminal or step == steps_per_epoch - 1:
                    # terminated by max episode steps
                    if not done:
                        last_v = self.actor.step(
                            torch.from_numpy(obs).type(torch.float32).to(
                                self.device))[1]
                    else:  # Agent terminated episode
                        last_v = 0

                    if terminal:
                        # only log these for terminals
                        eps_len_logs += [eps_len]
                        eps_ret_log += [eps_ret]

                    self.memory.end_eps(value=last_v)

                    obs = self.env.reset()
                    obs = self.convert_obs(obs)

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
            try:
                MaxEpsReturn = np.max(eps_ret_log)
                MinEpsReturn = np.min(eps_ret_log)
            except ValueError:
                MaxEpsReturn = 0
                MinEpsReturn = 0

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

            # Save model
            final_epoch = t == n_epochs - 1

            if (t and not t % self.args['save_frequency']) or final_epoch:
                print('Saving model..')
                self.save()
