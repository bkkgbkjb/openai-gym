from typing import Union, TypeVar, List, Generic, cast, Tuple, List
from utils.step import Step, NotNoneStep
from utils.transition import Transition
import numpy as np
from utils.common import AllowedStates, Action, Reward, Info
import math
from typing_extensions import Self

ES = TypeVar("ES", bound=Union[Step, NotNoneStep])
EpisodeGeneric = List[ES]

EPS = TypeVar("EPS", bound=AllowedStates)


class Episodes(Generic[EPS]):

    def __init__(self):
        self._steps: List[Step[EPS]] = []
        self.returns_computed = False
        self.advantage_computed = False
        self.clear()

    def get_step(self, i: int) -> Step[EPS]:
        return self.steps[i]

    def clear(self) -> Self:
        self._steps = []
        self.advantage_computed = False
        self.returns_computed = False
        return self

    @property
    def len(self) -> int:
        return 0 if len(self._steps) == 0 else len(self._steps) - 1

    @property
    def steps(self) -> List[Step[EPS]]:
        (_, la, lr, _) = self._steps[-2].details
        assert la is not None
        assert lr is not None and not math.isnan(lr)

        return self._steps[:-1]

    @property
    def non_stop_steps(self) -> List[NotNoneStep[EPS]]:
        return [
            NotNoneStep.from_step(s) for s in self.steps if s.is_not_none()
        ]

    def append_step(self, step: Step[EPS]) -> Self:
        (_, a, r, info) = step.details

        if len(self._steps) > 0:
            (_, la, _, _) = self._steps[-1].details
            if la is None:
                assert a is not None

        self._steps.append(step)

        return self

    @classmethod
    def from_list(cls, sari: Tuple[List[EPS], List[Action], List[Reward],
                                   List[Info]]):
        inst = cls()
        (s, a, r, info) = sari
        assert len(s) == len(a) + 1 == len(r) + 1 == len(info)

        for i in range(len(a)):

            inst.append_step(Step(s[i], a[i], r[i], info[i]))

        inst.append_step(Step(s[-1], None, None, info[-1]))

        return inst

    def get_steps(self, s_list: List[int]) -> List[Step[EPS]]:

        steps = self.steps
        return [steps[i] for i in s_list]

    def append_transition(self, transition: Transition[EPS]) -> Self:
        (s1, s2) = transition.as_tuple()
        (_, a, r, i1) = s1.details

        if len(self._steps) == 0:
            self._steps.extend([s1.to_step(), s2])
            return self

        (ls, la, lr, li) = self._steps[-1].details

        if not self._steps[-1].is_end():
            assert la is None
            assert lr is None
            assert li.items() <= i1.items()
            self._steps[-1] = Step(ls, a, r, i1)
            self._steps.append(s2)
            return self

        self._steps.extend([s1.to_step(), s2])

        return self

    def sample(self, batch_size: int) -> List[Step[EPS]]:
        idx = np.random.choice(self.len, batch_size)
        b = []
        for i in idx:
            b.append(self.steps[i])

        return b

    def cut(self,
            length: int,
            allow_last_not_align: bool = False
            ) -> List[List[NotNoneStep[EPS]]]:

        assert length >= 1

        steps = self.steps
        rl: List[List[NotNoneStep[EPS]]] = []
        l: List[NotNoneStep[EPS]] = []

        for s in steps:
            assert s.is_not_none()
            l.append(NotNoneStep.from_step(s))
            if len(l) == length:
                rl.append(l)
                l = []

        if allow_last_not_align and len(l) != 0:
            rl.append(l)

        return rl

    def sample_non_stop(self, batch_size: int) -> List[NotNoneStep[EPS]]:
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

        returns = [float("nan") for _ in range(l)]

        (_, a, _, i) = steps[-1].details

        next_is_stop = steps[-1].is_end()

        next_value = 0 if next_is_stop else i["value"]
        returns[-1] = next_value

        for j in reversed(range(l - 1)):
            (_, a, r, _) = steps[j].details
            if a is None:
                assert not next_is_stop
                next_is_stop = True
                next_value = 0
                returns[j] = 0
            else:
                assert r is not None
                returns[j] = r + (0 if next_is_stop else gamma * next_value)
                next_is_stop = False
                next_value = returns[j]

        for i, r in enumerate(returns):
            assert "return" not in steps[i].info
            steps[i].info["return"] = r

        self.returns_computed = True

    def compute_advantages(self, gamma: float = 0.99):
        if self.advantage_computed:
            return

        self.compute_returns(gamma)
        steps = self.steps
        l = self.len

        for i in range(l):
            (_, a, _, i) = steps[i].details
            assert "advantage" not in i
            i["advantage"] = i["return"] - (i["value"] if a is not None else 0)

        self.advantage_computed = True
