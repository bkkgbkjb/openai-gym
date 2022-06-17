import numpy as np
from typing import Dict, Tuple, Optional, Any, Union
from utils.algorithm import ActionInfo
import numpy as np
from utils.env_sb3 import LazyFrames
import torch

Action = np.ndarray

Reward = float

Info = Dict[str, Any]

SARSA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]

AllowedStates = Union[torch.Tensor, LazyFrames]
EmptyStates = Union[Optional[torch.Tensor], Optional[LazyFrames]]
AllAllowedStates = Union[AllowedStates, EmptyStates]