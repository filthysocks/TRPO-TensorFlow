import gym
from action_distribution import Continuous, Discret
from enum import Enum

class Environment(object):
    def __init__(self, config, env_name):
        self.animate = config.animate
        self.name = env_name
        self.env = gym.make(self.name)
        self.env.max_episode_steps = config.timestep_limit
        self.env._max_episode_steps = config.timestep_limit
        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self._get_action_dim()
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

    def _get_action_dim(self):
        raise NotImplementedError('_get_action needs to be implemented in child class')


class MountainCarContinuous(Environment):
    NAME = 'MountainCarContinuous-v0'

    def __init__(self, config):
        super(MountainCarContinuous, self).__init__(config, MountainCarContinuous.NAME)
        self._ad = Continuous(self.ac_dim, init_log_var=config.init_log_var)

    def _get_action_dim(self):
        return self.env.action_space.shape[0]

class MountainCar(Environment):
    NAME = 'MountainCar-v0'

    def __init__(self, config):
        super(MountainCar, self).__init__(config, MountainCar.NAME)
        cta = lambda x: x
        self._ad = Discret(self.ac_dim, class_to_action=cta)

    def _get_action_dim(self):
        return self.env.action_space.n

class CartPole(Environment):
    NAME = 'CartPole-v1'

    def __init__(self, config):
        super(CartPole, self).__init__(config, CartPole.NAME)
        cta = lambda x: x
        self._ad = Discret(self.ac_dim, class_to_action=cta)

    def _get_action_dim(self):
        return self.env.action_space.n

class Pendulum(Environment):
    NAME = 'Pendulum-v0'

    def __init__(self, config):
        super(Pendulum, self).__init__(config, Pendulum.NAME)
        self._ad = Continuous(self.ac_dim, init_log_var=config.init_log_var)

    def _get_action_dim(self):
        return self.env.action_space.shape[0]

class Environments:
    MOUNTAIN_CAR_CONTINUOUS = MountainCarContinuous
    MOUNTAIN_CAR = MountainCar
    CART_POLE = CartPole
    PENDULUM = Pendulum
