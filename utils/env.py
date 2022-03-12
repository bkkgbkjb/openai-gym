import gym
from typing import List, Tuple, Literal, Any, Optional, cast, Callable, Union, Iterable
from gym.wrappers import FrameStack, LazyFrames
from gym.spaces import Box
import numpy as np
from torchvision import transforms as T
import torch


def resolve_lazy_frames(lazy_frames: Any) -> torch.Tensor:
    assert len(lazy_frames) == 4
    rlt = torch.cat(cast(List[torch.Tensor], [lazy_frames[0], lazy_frames[1],
                    lazy_frames[2], lazy_frames[3]])).unsqueeze(0)
    assert rlt.shape == (1, 4, 84, 84)
    return rlt


class PreprocessObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = Box(
            low=0,
            high=1,
            shape=(1, 84, 84),
            dtype=np.float32,
        )

        self.transform = T.Compose(
            [T.ToPILImage(), T.Resize((84, 84)), T.Grayscale(), T.ToTensor()]
        )

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation


class ToTensorEnv(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        (h, w, c) = env.observation_space.shape

        self.observation_space = Box(
            low=0, high=1, dtype=np.float32, shape=(c, h, w))

        self.transform = T.ToTensor()

    def observation(self, observation):
        observation = self.transform(observation)
        assert observation.shape == self.observation_space.shape
        return observation
