from typing import Dict, Generic, Tuple, TypeVar, Optional, List, Any, Union, cast
import numpy as np
from utils.env_sb3 import LazyFrames, resolve_lazy_frames
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Action = np.ndarray
ActionInfo = Tuple[Action, Dict[str, Any]]

Reward = float

AllowedStates = TypeVar('AllowedStates', bound=Union[torch.Tensor, LazyFrames])

AllAllowedStates = TypeVar('AllAllowedStates',
                           bound=Union[torch.Tensor, Optional[torch.Tensor],
                                       LazyFrames, Optional[LazyFrames]])
A = TypeVar('A')
R = TypeVar('R')
BaseStep = Tuple[AllAllowedStates, A, R]

Step = BaseStep[AllowedStates, Optional[ActionInfo], Optional[Reward]]

NotNoneStep = BaseStep[AllowedStates, ActionInfo, Reward]

AllNoneStep = BaseStep[AllAllowedStates, Optional[ActionInfo],
                       Optional[Reward]]

ES = TypeVar("ES", bound=Union[Step, NotNoneStep, AllNoneStep])
EpisodeGeneric = List[ES]

SARSA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor]

TransitionTuple = Tuple[AllowedStates, ActionInfo, Reward, AllowedStates,
                        Union[Optional[ActionInfo], bool]]


class Transition(Generic[AllowedStates]):

    def __init__(self, sarsa: TransitionTuple[AllowedStates]):
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

    def as_tuple(self) -> TransitionTuple[AllowedStates]:
        return (self.s, self.a, self.r, self.sn, self.an)


def resolve_transitions(trs: List[Transition[AllowedStates]],
                        state_shape: Tuple, action_shape: Tuple) -> SARSA:
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
