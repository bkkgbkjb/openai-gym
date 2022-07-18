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


class Episode(Generic[EPS]):
    def __init__(self):
        self._steps: List[Step[EPS]] = []
        self.returns_computed = False
        self.advantage_computed = False
        self._end = False

    @property
    def steps(self) -> List[NotNoneStep[EPS]]:
        (_, la, lr, _) = self._steps[-2].details
        assert la is not None
        assert lr is not None

        return [NotNoneStep.from_step(s) for s in self._steps[:-1]]

    def __getitem__(self, idx: int) -> NotNoneStep:
        assert 1 - len(self._steps) <= idx < len(self._steps) - 1

        return NotNoneStep.from_step(self._steps[idx if idx >= 0 else idx - 1])
    
    @property
    def last_state(self) -> EPS:
        return self._steps[-1].state

    @property
    def end(self) -> bool:
        return self._end

    @property
    def len(self) -> int:
        return 0 if len(self._steps) == 0 else len(self._steps) - 1

    def append_step(self, step: Step[EPS]) -> Self:
        assert not self.end, "cannot append step into ended episode"
        (_, a, r, info) = step.details

        if len(self._steps) > 0:
            (_, la, _, li) = self._steps[-1].details
            if la is None:
                assert a is not None
            assert "next" not in li
            li["next"] = step

        self._steps.append(step)
        if step.is_end():
            assert step.action is None
            assert step.reward is None
            self._end = True

        return self

    @classmethod
    def from_list(cls, sari: Tuple[List[EPS], List[Action], List[Reward], List[Info]]):
        (s, a, r, info) = sari
        assert info[-1]["end"]

        inst = cls()
        assert len(s) == len(a) + 1 == len(r) + 1 == len(info)

        for i in range(len(a)):

            inst.append_step(Step(s[i], a[i], r[i], info[i]))

        inst.append_step(Step(s[-1], None, None, info[-1]))

        return inst

    def get_steps(self, s_list: List[int]) -> List[NotNoneStep[EPS]]:

        steps = self.steps
        return [steps[i] for i in s_list]

    def append_transition(self, transition: Transition[EPS]) -> Self:
        assert not self.end, "cannot append transition into a ended episode"
        (s1, s2) = transition.as_tuple()
        (_, a, r, i1) = s1.details

        if len(self._steps) == 0:
            _s = s1.to_step()
            assert "next" not in _s.info
            _s.info["next"] = s2
            self._steps.extend([_s, s2])
            return self

        (ls, la, lr, li) = self._steps[-1].details

        assert la is None
        assert lr is None
        assert li.items() <= i1.items()

        assert "next" not in i1
        i1["next"] = s2
        self._steps[-1] = Step(ls, a, r, i1)
        self._steps.append(s2)
        return self

    def sample(self, batch_size: int) -> List[Step[EPS]]:
        idx = np.random.choice(self.len, batch_size)
        b = []
        for i in idx:
            b.append(self._steps[i])

        return b

    @classmethod
    def cut(
        cls,
        episode: Self,
        length: int,
        allow_last_not_align: bool = False,
        start: int = 0,
    ) -> List[Self]:

        assert length >= 1

        assert 0 <= start < length

        re: List[Self] = []
        e: Self = cls()

        steps = episode.steps[start:]
        for s in steps:
            del s.info['next']
            e.append_step(s.to_step())
            if e.len == length:
                re.append(e)
                e = cls()

        if allow_last_not_align and e.len != 0:
            re.append(e)

        return re

    def compute_returns(self, gamma: float = 0.99):
        if self.returns_computed:
            return

        steps = self.steps

        rwd = 0
        for s in reversed(steps):
            assert "return" not in s.info
            rwd = s.info["return"] = gamma * rwd + s.reward

        self.returns_computed = True

    def compute_advantages(self, gamma: float = 0.99):
        if self.advantage_computed:
            return

        self.compute_returns(gamma)
        steps = self.steps
        l = self.len

        for i in range(l):
            s = steps[i]
            (_, a, _, i) = s.details
            assert "advantage" not in i

            # TODO: 是不是该用value - return ？
            i["advantage"] = i["return"] - (i["value"] if not s.is_end() else 0)

        self.advantage_computed = True
