# following code is manually copied from stable-baseline3, as it depends on an older version of gym, but I don't want to solve this version conflict

import gym
import numpy as np
from gym import spaces
from collections import deque
from typing import Optional, Any, cast, List, Union
import torch
import numpy as np
from gym.spaces import Box
from gym import ObservationWrapper

try:
    import cv2  # pytype:disable=import-error

    cv2.ocl.setUseOpenCL(False)
except ImportError:
    cv2 = None


class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.

    :param env: the environment
    :param width:
    :param height:
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=env.observation_space.dtype,
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clips the reward to {+1, 0, -1} by its sign.

    :param env: the environment
    """

    def __init__(self, env: gym.Env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward: float) -> float:
        """
        Bin reward to {+1, 0, -1} by its sign.

        :param reward:
        :return:
        """
        return np.sign(reward)


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every ``skip``-th frame (frameskipping)

    :param env: the environment
    :param skip: number of ``skip``-th frame
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2, ) + env.observation_space.shape,
                                    dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action: int):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SkipFrames(gym.Wrapper):
    """Skip timesteps: repeat action, accumulate reward, take last obs."""

    def __init__(self, env, skip: int = 4):
        super(SkipFrames, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        for i in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            info["steps"] = i + 1
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.

    :param env: the environment to wrap
    :param noop_max: the maximum value of no-ops to run
    """

    def __init__(self, env: gym.Env, noop_max: int = 30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = np.zeros(0)
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.

    :param env: the environment to wrap
    """

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset, only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.

        :param kwargs: Extra keywords passed to env.reset() call
        :return: the first observation of the environment
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class LazyFrames:
    r"""Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.

    .. note::

        This object should only be converted to numpy array just before forward pass.

    Args:
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames), ) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(
                self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]],
            axis=0)

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame),
                                 dtype=self.dtype).reshape(self.frame_shape)
        return frame


class FrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.

    Example::

        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)

    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally

    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...],
                        num_stack,
                        axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...],
                         num_stack,
                         axis=0)
        self.observation_space = Box(low=low,
                                     high=high,
                                     dtype=self.observation_space.dtype)

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames),
                                                    self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)
            info = None  # Unused
        [self.frames.append(obs) for _ in range(self.num_stack)]

        if kwargs.get("return_info", False):
            return self.observation(), info
        else:
            return self.observation()


def resolve_lazy_frames(lazy_frames: LazyFrames) -> torch.Tensor:

    rlt = torch.cat(
        cast(
            List[torch.Tensor],
            [lazy_frames[i] for i in range(len(lazy_frames))],
        ))

    assert rlt.size(0) == len(lazy_frames)
    return rlt
