import numpy as np
from typing import Dict, Generic, Iterable, Tuple, TypeVar, Optional, List, Any, Union, cast
from typing_extensions import Self
import math
import numpy as np
from utils.env_sb3 import LazyFrames, resolve_lazy_frames
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Action = np.ndarray
ActionInfo = Tuple[Action, Dict[str, Any]]

Reward = float

AllowedStates = Union[torch.Tensor, LazyFrames]
EmptyStates = Union[Optional[torch.Tensor], Optional[LazyFrames]]
AllAllowedStates = Union[AllowedStates, EmptyStates]

S = TypeVar('S', bound=AllAllowedStates)
A = TypeVar('A')
R = TypeVar('R')
BaseStep = Tuple[S, A, R]

SS = TypeVar('SS', bound=AllowedStates)
Step = BaseStep[SS, Optional[ActionInfo], Optional[Reward]]

NNS = TypeVar('NNS', bound=AllowedStates)
NotNoneStep = BaseStep[NNS, ActionInfo, Reward]

ANS = TypeVar('ANS', bound=EmptyStates)
AllNoneStep = BaseStep[ANS, Optional[ActionInfo], Optional[Reward]]

ES = TypeVar("ES", bound=Union[Step, NotNoneStep, AllNoneStep])
EpisodeGeneric = List[ES]

SARSA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]

TTS = TypeVar('TTS', bound=AllowedStates)
TransitionTuple = Tuple[TTS, ActionInfo, Reward, TTS,
                        Union[Optional[ActionInfo], bool]]

TS = TypeVar("TS", bound=AllowedStates)


class Transition(Generic[TS]):

    def __init__(self, sarsa: TransitionTuple[TS]):
        (s, a, r, sn, an) = sarsa

        assert an is None or isinstance(an, bool) or isinstance(an, tuple)

        self.state_shape = s.shape
        self.action_shape = a[0].shape

        self.s = s
        self.a = a
        self.r = r
        self.sn = sn
        self.an = an

        assert self.s.shape == self.sn.shape
        if isinstance(self.an, tuple):
            assert self.a[0].shape == self.an[0].shape

    def as_tuple(self) -> TransitionTuple[TS]:
        return (self.s, self.a, self.r, self.sn, self.an)


RTS = TypeVar('RTS', bound=AllowedStates)


def resolve_transitions(trs: List[Transition[RTS]], state_shape: Tuple,
                        action_shape: Tuple) -> SARSA:
    l = len(trs)
    trs_tuples = [trs.as_tuple() for trs in trs]

    states = torch.stack([
        s if isinstance(s, torch.Tensor) else resolve_lazy_frames(s)
        for (s, _, _, _, _) in trs_tuples
    ])
    assert states.shape == ((l, ) + state_shape)
    assert not states.requires_grad

    actions = torch.stack([
        torch.from_numpy(a).type(torch.float32)
        if a.dtype == np.float64 else torch.from_numpy(a)
        for (_, (a, _), _, _, _) in trs_tuples
    ])

    assert actions.shape == ((l, ) + action_shape)
    assert not states.requires_grad

    rewards = torch.stack([
        torch.tensor(r, dtype=torch.float32) for (_, _, r, _, _) in trs_tuples
    ]).unsqueeze(1)

    assert rewards.shape == (l, 1)
    assert not rewards.requires_grad

    next_states = torch.stack([
        sn if isinstance(sn, torch.Tensor) else resolve_lazy_frames(sn)
        for (_, _, _, sn, _) in trs_tuples
    ])

    assert next_states.shape == ((l, ) + state_shape)
    assert not next_states.requires_grad

    done = torch.as_tensor(
        [
            1 if (an is None or an == True) else 0
            for (_, _, _, _, an) in trs_tuples
        ],
        dtype=torch.int8,
    ).unsqueeze(1)

    assert done.shape == (l, 1)
    assert not done.requires_grad

    return (
        states.to(DEVICE),
        actions.to(DEVICE),
        rewards.to(DEVICE),
        next_states.to(DEVICE),
        done.to(DEVICE),
    )


EPS = TypeVar('EPS', bound=AllowedStates)


class Episodes(Generic[EPS]):

    def __init__(self):
        self._steps: List[Tuple[Step[EPS], Dict[str, Any]]] = []
        self.returns_computed = False
        self.advantage_computed = False
        self.gae_computed = False

    @property
    def len(self) -> int:
        return len(self._steps) - 1

    @property
    def steps(self) -> List[Tuple[Step[EPS], Dict[str, Any]]]:
        ((_, la, lr), _) = self._steps[-2]
        assert la is not None
        assert lr is not None and not math.isnan(lr)

        return self._steps[:-1]

    @property
    def pure_steps(self) -> List[Step[EPS]]:
        return [s for (s, _) in self.steps]

    @property
    def non_stop_steps(self) -> List[Tuple[NotNoneStep[EPS], Dict[str, Any]]]:
        return [((s, a, cast(Reward, r)), info)
                for ((s, a, r), info) in self.steps if a is not None]

    def append_step(self, step: Step[EPS]) -> Self:
        (_, a, r) = step
        if a is None:
            assert r is None

        if len(self._steps) > 0:
            ((_, la, _), _) = self._steps[-1]
            if la is None:
                assert a is not None

        self._steps.append((step, dict()))

        return self

    def append_transition(self, transition: Transition[EPS]) -> Self:
        (s, a, r, sn, an) = transition.as_tuple()
        end = False

        if an is None or an == True:
            end = True

        if len(self._steps) == 0:
            self._steps.extend([((s, a, r), dict()),
                                ((sn, None, None if end else float('nan')),
                                 dict())])
            return self

        ((ls, la, lr), li) = self._steps[-1]

        if lr is not None:
            assert math.isnan(lr)

        if lr is not None and math.isnan(lr):
            assert la is None
            self._steps[-1] = ((ls, a, r), li)
            self._steps.append(
                ((sn, None, None if end else float('nan')), dict()))
            return self

        self._steps.extend([((s, a, r), dict()),
                            ((sn, None, None if end else float('nan')), dict())
                            ])

        return self

    def clear(self) -> Self:
        self._steps = []
        self.advantage_computed = False
        self.returns_computed = False
        self.gae_computed = False
        return self

    def sample(self,
               batch_size: int) -> List[Tuple[Step[EPS], Dict[str, Any]]]:
        idx = np.random.choice(self.len, batch_size)
        b = []
        for i in idx:
            b.append(self.steps[i])

        return b

    def sample_non_stop(
            self,
            batch_size: int) -> List[Tuple[NotNoneStep[EPS], Dict[str, Any]]]:
        idx = np.random.choice(len(self.non_stop_steps), batch_size)
        b = []
        for i in idx:
            b.append(self.non_stop_steps[i])

        return b

    def compute_returns(self, gamma: float = 0.99):
        if self.returns_computed:
            return

        steps = self.steps
        l = self.len

        returns = [float('nan') for _ in range(l)]

        ((_, a, _), _) = steps[-1]

        next_is_stop = a is None

        next_value = 0 if next_is_stop else a[1]['value']
        returns[-1] = next_value

        for j in reversed(range(l - 1)):
            ((_, a, r), _) = steps[j]
            if a is None:
                assert not next_is_stop
                next_is_stop = True
                next_value = 0
                returns[j] = 0
            else:
                assert r is not None
                returns[j] = (r + (0 if next_is_stop else gamma * next_value))
                next_is_stop = False
                next_value = returns[j]

        for i, r in enumerate(returns):
            steps[i][1]['return'] = r

        self.returns_computed = True

    def compute_advantages(self, gamma: float = 0.99):
        if self.advantage_computed:
            return

        self.compute_returns(gamma)
        steps = self.steps
        l = self.len

        for i in range(l):
            ((_, a, _), i) = steps[i]
            i['advantage'] = i['return'] - (a[1]['value']
                                            if a is not None else 0)

        self.advantage_computed = True
