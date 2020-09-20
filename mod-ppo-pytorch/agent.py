"""
    Grid2op  Agent
"""
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from ppo import PPO


class MLPAgent(AgentWithConverter):
    def __init__(self, env, observation_space, action_space, ppo_args):
        super(MLPAgent, self).__init__(action_space,
                                       action_space_converter=IdToAct)
        self.obs_space = observation_space

        print('Filtering actions..')
        self.action_space.filter_action(self._filter_act)
        print('Done')

        self.ppo_agent = PPO(env, **ppo_args)

        self.obs_size = self.obs_space.size()
        self.action_space = self.action_space.size()

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

    def my_act(self, transformed_obs, reward=None, done=False):
        """
            Used by the agent to decide on action to take

            Returns an `encoded_action` which is reconverted
            by the inherited `self.convert_act` into a valid
            action that can be taken in the env


        """

        act = self.ppo_agent.predict_action(transformed_obs)

        return act

    def load(self, path):
        """
            Loads a trained model from the given path
        """
        self.ppo_agent.load_actor(path)

    def save(self, path):
        """
            Saves model params
        """
        self.ppo_agent.save_actor(path)
