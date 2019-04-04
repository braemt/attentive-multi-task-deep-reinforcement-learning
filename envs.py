import gym
import logging
import gym_pycolab
import universe
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()


def create_env(env_id):
    return create_pycolab_env(env_id)

def create_pycolab_env(env_id):
    env = gym.make(env_id)
    env = Vectorize(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env


def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)


class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503, *args, **kwargs):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._last_time_in_episode = time.time()
        self._fps_in_episode = 0
        self._frames_in_episode = 0
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._episode_actions = []
        self._episode_count = 0
        self._last_episode_id = -1

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1

        cur_episode_id = info.get('vectorized.episode_id', 0)
        if self._last_episode_id == cur_episode_id:
            self._frames_in_episode += 1
            cur_time = time.time()
            elapsed = max(1e-5, cur_time - self._last_time_in_episode)
            self._fps_in_episode = self._frames_in_episode / float(elapsed)
        else:
            self._frames_in_episode = 0
            self._last_episode_id = cur_episode_id
            self._last_time_in_episode = time.time()

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = max(1e-5, cur_time - self._last_time)
            fps = self._log_interval / float(elapsed)
            self._last_time = cur_time
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = self._fps_in_episode
            self._last_episode_id = cur_episode_id

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = max(1e-5, time.time() - self._episode_time)
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / float(total_time)

            self._episode_reward = 0
            self._episode_length = 0
            self._episode_actions = []
            self._episode_count += 1

        return observation, reward, done, to_log