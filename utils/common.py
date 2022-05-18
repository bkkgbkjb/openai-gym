from typing import Dict, Tuple, TypeVar, Optional, List, Any
import numpy as np

Action = np.ndarray
ActionInfo = Tuple[Action, Dict[str, Any]]

Reward = float

OS = TypeVar("OS")
StepGeneric = Tuple[OS, Optional[ActionInfo], Optional[Reward]]

NOS = TypeVar("NOS")
NotNoneStepGeneric = Tuple[NOS, ActionInfo, Reward]


OE = TypeVar("OE")
EpisodeGeneric = List[StepGeneric[OE]]


TS = TypeVar("TS")
TransitionGeneric = Tuple[TS, ActionInfo, Reward, TS, Optional[ActionInfo]]
