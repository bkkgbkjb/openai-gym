from typing import Dict, Tuple, TypeVar, Optional, List, Any


OS = TypeVar("OS")
AS = TypeVar("AS")
StepGeneric = Tuple[OS, Optional[AS], Optional[float]]

NOS = TypeVar("NOS")
NAS = TypeVar("NAS")
NotNoneStepGeneric = Tuple[NOS, NAS, float]


OE = TypeVar("OE")
AE = TypeVar("AE")
Episode = List[StepGeneric[OE, AE]]

AA = TypeVar("AA")
ActionInfo = Tuple[AA, Dict[str, Any]]

TS = TypeVar("TS")
TA = TypeVar("TA")
TransitionGeneric = Tuple[TS, ActionInfo[TA],
                          float, TS, Optional[ActionInfo[TA]]]
