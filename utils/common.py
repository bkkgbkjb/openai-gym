import numpy as np
from typing import Dict, Tuple, Optional, Any, Union, List
import numpy as np
from utils.env_sb3 import LazyFrames
import torch

Action = torch.Tensor
ActionInfo = Tuple[Action, Dict[str, Any]]

Reward = float

Info = Dict[str, Any]

SARSAI = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, List[Info]]

AllowedStates = Union[torch.Tensor, LazyFrames]
EmptyStates = Union[Optional[torch.Tensor], Optional[LazyFrames]]
AllAllowedStates = Union[AllowedStates, EmptyStates]