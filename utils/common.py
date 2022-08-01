import numpy as np
from typing import Dict, Tuple, Optional, Any, Union, List, TypeVar, cast
import numpy as np
import torch

class LazyFrames:
    r"""Ensures common frames are only stored once to optimize memory use.

    To further reduce the memory use, it is optionally to turn on lz4 to
    compress the observations.

    .. note::

        This object should only be converted to numpy array just before forward pass.

    Args:
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames, lz4_compress=False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames), ) + self.frame_shape
        self.dtype = frames[0].dtype
        if lz4_compress:
            from lz4.block import compress

            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(
                self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]],
            axis=0)

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame),
                                 dtype=self.dtype).reshape(self.frame_shape)
        return frame

Action = torch.Tensor
ActionInfo = Tuple[Action, Dict[str, Any]]

Reward = float

Info = Dict[str, Any]

SARSAI = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
              torch.Tensor, List[Info]]

AllowedStates = Union[torch.Tensor, LazyFrames]
EmptyStates = Union[Optional[torch.Tensor], Optional[LazyFrames]]
AllAllowedStates = Union[AllowedStates, EmptyStates]

S = TypeVar("S")

def resolve_lazy_frames(lazy_frames: LazyFrames) -> torch.Tensor:

    rlt = torch.stack(
        cast(
            List[torch.Tensor],
            [lazy_frames[i] for i in range(len(lazy_frames))],
        ))

    assert rlt.size(0) == len(lazy_frames)
    return rlt

ActionScale = Union[float, torch.Tensor]