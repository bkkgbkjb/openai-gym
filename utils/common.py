from typing import Dict, Tuple, TypeVar, Optional, List, Any, Union
import numpy as np
from utils.env_sb3 import LazyFrames
import torch

Action = np.ndarray
ActionInfo = Tuple[Action, Dict[str, Any]]

Reward = float

AllowedState = Union[torch.Tensor, LazyFrames]

OS = TypeVar("OS", bound=AllowedState)
StepGeneric = Tuple[OS, Optional[ActionInfo], Optional[Reward]]

NOS = TypeVar("NOS", bound=AllowedState)
NotNoneStepGeneric = Tuple[NOS, ActionInfo, Reward]

OE = TypeVar("OE", bound=AllowedState)
EpisodeGeneric = List[StepGeneric[OE]]

TS = TypeVar("TS", bound=AllowedState)
TransitionGeneric = Tuple[TS, ActionInfo, Reward, TS, Optional[ActionInfo]]
