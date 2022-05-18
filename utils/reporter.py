from typing import Dict, Any
from torch.utils.tensorboard import SummaryWriter


def get_reporter(name: str):
    writer = SummaryWriter(comment='_' + name)
    times_counter: Dict[str, int] = dict()

    def reporter(info: Dict[str, Any]):
        nonlocal times_counter

        for k, v in info.items():
            if not k in times_counter:
                times_counter[k] = 0
            writer.add_scalar(k, v, times_counter[k])
            times_counter[k] = times_counter[k] + 1

    return reporter
