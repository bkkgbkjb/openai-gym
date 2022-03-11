from typing import Dict, Tuple, TypeVar, Optional, List


OS = TypeVar("OS")
AS = TypeVar("AS")
Step = Tuple[OS, Optional[AS], Optional[float]]


OE = TypeVar("OE")
AE = TypeVar("AE")
Episode = List[Step[OE, AE]]

AA = TypeVar("AA")
ActionInfo = Tuple[AA, Dict]

TS = TypeVar("TS")
TA = TypeVar("TA")
TransitionGeneric = Tuple[TS, ActionInfo[TA], float, TS, Optional[ActionInfo[TA]]]
