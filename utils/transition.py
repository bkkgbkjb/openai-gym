from typing import Generic, Tuple, TypeVar, List
from utils.common import AllowedStates, SARSA
import torch
from utils.step import NotNoneStep, Step
from utils.env_sb3 import resolve_lazy_frames
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TTS = TypeVar('TTS', bound=AllowedStates)
TransitionTuple = Tuple[NotNoneStep[TTS], Step[TTS]]

TS = TypeVar("TS", bound=AllowedStates)


class Transition(Generic[TS]):

    def __init__(self, sarsa: TransitionTuple[TS]):
        (step1, step2) = sarsa
        (s, a, r, i1) = step1.details
        (sn, an, rn, i2) = step2.details

        if step2.is_end():
            assert an is None
            assert rn is None

        self.state_shape = s.shape
        self.action_shape = a.shape

        self.s = s
        self.a = a
        self.r = r
        self.sn = sn
        self.an = an
        self.rn = rn
        self.i1 = i1
        self.i2 = i2

        self.step1 = step1
        self.step2 = step2

        assert self.s.shape == self.sn.shape
        if isinstance(self.an, np.ndarray):
            assert self.a.shape == self.an.shape

    def as_tuple(self) -> TransitionTuple[TS]:
        return (self.step1, self.step2)

    def is_end(self) -> bool:
        return self.step2.is_end()


RTS = TypeVar('RTS', bound=AllowedStates)


def resolve_transitions(trs: List[Transition[RTS]], state_shape: Tuple,
                        action_shape: Tuple) -> SARSA:
    l = len(trs)
    trs_tuples = [trs.as_tuple() for trs in trs]

    states = torch.stack([
        s1.state if isinstance(s1.state, torch.Tensor) else
        resolve_lazy_frames(s1.state) for (s1, _) in trs_tuples
    ])
    assert states.shape == ((l, ) + state_shape)
    assert not states.requires_grad

    actions = torch.stack([
        s1.action
        for (s1, _) in trs_tuples
    ])

    assert actions.shape == ((l, ) + action_shape)
    assert not states.requires_grad

    rewards = torch.stack([
        torch.tensor(s1.reward, dtype=torch.float32) for (s1, _) in trs_tuples
    ]).unsqueeze(1)

    assert rewards.shape == (l, 1)
    assert not rewards.requires_grad

    next_states = torch.stack([
        s2.state if isinstance(s2.state, torch.Tensor) else
        resolve_lazy_frames(s2.state) for (_, s2) in trs_tuples
    ])

    assert next_states.shape == ((l, ) + state_shape)
    assert not next_states.requires_grad

    done = torch.as_tensor(
        [1 if s2.is_end() else 0 for (_, s2) in trs_tuples],
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
