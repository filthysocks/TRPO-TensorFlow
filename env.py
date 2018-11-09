import gym
from actioni_dstribution import Continuous
from enum import Enum


class Environment(object):
    def __init__(self, config, env_name):
        self.animate = config.animate
        self.name = env_name
        self.env = gym.make(self.name)
        self.env.max_episode_steps = config.timestep_limit
        # Make sure it is not a discrete env
        assert not isinstance(self.env.action_space, gym.spaces.Discrete)
        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.shape[0]
        self._ad = None

    def reset(self):
        s = self.env.reset()
        return s

    def step(self, action, animate=None):
        if animate is None:
            animate = self.animate
        s2, r, terminal, details = self.env.step(action)
        if animate:
            self.env.render()
        return s2, r, terminal, details

    @property
    def action_distribution(self):
        return self._ad


class MountainCarContinuous(Environment):
    NAME = 'MountainCarContinuous-v0'

    def __init__(self, config):
        super(MountainCarContinuous, self).__init__(config, MountainCarContinuous.NAME)
        self._ad = Continuous(self.ac_dim, init_log_var=config.init_log_var)


class Environments:
    MOUNTAIN_CAR_CONTINUOUS = MountainCarContinuous
