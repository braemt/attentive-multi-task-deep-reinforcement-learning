# THIS FILE IS NEW OR MODIFIED COMPARED TO https://github.com/deepmind/pycolab

import gym
import numpy as np
from gym import spaces

import gym_pycolab.examples.aperture as aperture


class PycolabEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    episode_id = 0

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self._observation_shape = [11, 14, 1]
        self.observation_space = spaces.Box(0.0, 1.0, self._observation_shape)
        self.game = self._make_game()

    def step(self, action):
        observation, reward, discount = self.game.play(actions=action)
        if reward is None:
            reward = 0.0
        game_over = (discount == 0.0)
        state = self._format_observation(observation.board)
        self.state = self._concat_obs(state)
        info = {'vectorized.episode_id': self.episode_id}
        return self.state, reward, game_over, info

    def reset(self):
        self.episode_id += 1
        self.game = self._make_game()
        observation, reward, discount = self.game.its_showtime()
        state = self._format_observation(observation.board)
        self.state = self._init_obs(state)
        return self.state

    def render(self, mode='human', close=False):
        return

    def _format_observation(self, state):
        shape = list(self._observation_shape)
        shape[2] = 1
        return np.reshape(state / 127.0, shape)

    def _make_game(self):
        return aperture.make_game(0)

    def _init_obs(self, init_state):
        init_list = []
        for i in range(self._observation_shape[2]):
            init_list.append(init_state)
        return np.concatenate(tuple(init_list), axis=-1)

    def _concat_obs(self, state):
        self.state = self.state[:, :, 1:]
        return np.concatenate((self.state, state), axis=-1)
