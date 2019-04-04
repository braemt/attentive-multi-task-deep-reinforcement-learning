# THIS FILE IS NEW OR MODIFIED COMPARED TO https://github.com/deepmind/pycolab

import random
import numpy as np

from gym import spaces

from gym_pycolab.envs.pycolab_env import PycolabEnv
from gym_pycolab.worlds import grid_worlds as grid_worlds
from gym_pycolab.worlds import connect_four

WALL_LEVEL = ['    #   ',
              '    #   ',
              '##  #   ',
              '        ',
              '        ',
              '   #  ##',
              '   #    ',
              '   #    ']


class PycolabConnectNEnv(PycolabEnv):
    def __init__(self):
        super(PycolabEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return connect_four.make_game(shape=self._shape, connect_n=6)

    def _setup(self):
        self.action_space = spaces.Discrete(self._shape[1])
        self._observation_shape = [self._shape[0], self._shape[1], 1]
        self.observation_space = spaces.Box(0.0, 1.0, self._observation_shape)


class PycolabConnect4Env(PycolabConnectNEnv):
    def __init__(self):
        self._shape = [8, 8]
        super(PycolabConnectNEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return connect_four.make_game(shape=self._shape, connect_n=4)


class PycolabConnect4RotatedEnv(PycolabConnectNEnv):
    def __init__(self):
        self._shape = [8, 8]
        super(PycolabConnectNEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return connect_four.make_game(shape=self._shape, connect_n=4)

    def _format_observation(self, state):
        state = super(PycolabConnect4RotatedEnv, self)._format_observation(state)
        return np.rot90(state, k=1)


class PycolabConnect5Env(PycolabConnectNEnv):
    def __init__(self):
        self._shape = [8, 8]
        super(PycolabConnectNEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return connect_four.make_game(shape=self._shape, connect_n=5)


class PycolabConnect5RotatedEnv(PycolabConnectNEnv):
    def __init__(self):
        self._shape = [8, 8]
        super(PycolabConnectNEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return connect_four.make_game(shape=self._shape, connect_n=5)

    def _format_observation(self, state):
        state = super(PycolabConnect5RotatedEnv, self)._format_observation(state)
        return np.rot90(state, k=1)


class PycolabGridWorldsEnv(PycolabEnv):
    def __init__(self):
        super(PycolabEnv, self).__init__()
        self._setup()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=None, terminal_reward=0.5, bonus_reward=0.5, per_step_cost=0.0,
                                     positive_rewards=0, negative_rewards=0, swap_rewards=False,
                                     confined_to_board=False, off_board_cost=0.5, wall_is_terminal=False, wall_cost=0.0,
                                     swap_actions=False, doors=False, round_world=False)

    def _setup(self):
        self.action_space = spaces.Discrete(4)
        self._observation_shape = [8, 8, 1]
        self.observation_space = spaces.Box(0.0, 1.0, self._observation_shape)


class PycolabGridWorldsLevel1Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(terminal_reward=1.0, per_step_cost=0.01, confined_to_board=True,
                                     off_board_cost=0.0)


class PycolabGridWorldsLevel2Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(terminal_reward=1.0)


class PycolabGridWorldsLevel3Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(positive_rewards=1)


class PycolabGridWorldsLevel4Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(positive_rewards=1, negative_rewards=1)


class PycolabGridWorldsLevel5Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(terminal_reward=0.4, positive_rewards=3, bonus_reward=0.2)


class PycolabGridWorldsLevel6Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(terminal_reward=0.4, positive_rewards=3, negative_rewards=1, bonus_reward=0.2)


class PycolabGridWorldsLevel7Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(positive_rewards=1, negative_rewards=1, swap_rewards=True)


class PycolabGridWorldsLevel8Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(terminal_reward=1.0, positive_rewards=3, swap_rewards=True, bonus_reward=0.2)


class PycolabGridWorldsLevel9Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(terminal_reward=0.4, negative_rewards=3, swap_rewards=True, bonus_reward=0.2)


class PycolabGridWorldsLevel10Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(terminal_reward=0.4, positive_rewards=1, negative_rewards=3, swap_rewards=True,
                                     bonus_reward=0.2)


class PycolabGridWorldsLevel11Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, terminal_reward=1.0, per_step_cost=0.01,
                                     confined_to_board=True, off_board_cost=0.0)


class PycolabGridWorldsLevel12Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, terminal_reward=1.0)


class PycolabGridWorldsLevel13Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, positive_rewards=1)


class PycolabGridWorldsLevel14Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, positive_rewards=1, negative_rewards=1)


class PycolabGridWorldsLevel15Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, terminal_reward=0.4, positive_rewards=3, bonus_reward=0.2)


class PycolabGridWorldsLevel16Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, terminal_reward=0.4, positive_rewards=3, negative_rewards=1,
                                     bonus_reward=0.2)


class PycolabGridWorldsLevel17Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, positive_rewards=1, negative_rewards=1, swap_rewards=True)


class PycolabGridWorldsLevel18Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, terminal_reward=1.0, positive_rewards=3, swap_rewards=True,
                                     bonus_reward=0.2)


class PycolabGridWorldsLevel19Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, terminal_reward=0.4, negative_rewards=3, swap_rewards=True,
                                     bonus_reward=0.2)


class PycolabGridWorldsLevel20Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(raw_level=WALL_LEVEL, terminal_reward=0.4, positive_rewards=1, negative_rewards=3,
                                     swap_rewards=True, bonus_reward=0.2)


class PycolabGridWorldsLevel101Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(terminal_reward=1.0, swap_actions=True)


class PycolabGridWorldsLevel102Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(positive_rewards=1, swap_actions=True)


class PycolabGridWorldsLevel103Env(PycolabGridWorldsEnv):
    def __init__(self):
        super(PycolabGridWorldsEnv, self).__init__()

    def _make_game(self):
        self._setup()
        return grid_worlds.make_game(positive_rewards=1, negative_rewards=1, swap_actions=True)