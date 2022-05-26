from typing import Dict, Tuple, TypeVar, Optional, List, Any, Union
import numpy as np
from utils.agent import Agent, OfflineAgent
from utils.env_sb3 import LazyFrames
import torch

Action = np.ndarray
ActionInfo = Tuple[Action, Dict[str, Any]]

Reward = float

AllowedState = Union[torch.Tensor, LazyFrames]

S = TypeVar('S')
A = TypeVar('A')
R = TypeVar('R')
BaseStep = Tuple[S, A, R]

Step = BaseStep[AllowedState, Optional[ActionInfo], Optional[Reward]]

NotNoneStep = BaseStep[AllowedState, ActionInfo, Reward]

AllNoneStep = BaseStep[Optional[AllowedState], Optional[ActionInfo],
                       Optional[Reward]]

ES = TypeVar("ES", bound=Union[Step, NotNoneStep, AllNoneStep])
EpisodeGeneric = List[ES]

Transition = Tuple[AllowedState, ActionInfo, Reward, AllowedState,
                   Union[Optional[ActionInfo], bool]]

Observation = TypeVar('Observation')

O = TypeVar('O')
AllAgent = Union[Agent[O], OfflineAgent[O]]
