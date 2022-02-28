from optparse import Option
from typing import Tuple, TypeVar, Optional, List


OS = TypeVar('OS')
AS = TypeVar('AS')
Step = Tuple[OS, Optional[AS], Optional[float]]


OE = TypeVar('OE')
AE = TypeVar('AE')
Episode = List[Step[OE, AE]]

TS = TypeVar('TS')
TA = TypeVar('TA')
TransitionGeneric = Tuple[TS, TA, float, TS, Optional[TA], Optional[float]]
