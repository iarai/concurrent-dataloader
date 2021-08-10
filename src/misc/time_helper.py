import time
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Union

import numpy as np

PRECISION = 6


class TimeHelper:
    def __init__(self) -> None:
        self.recordings = defaultdict(list)

    def record(self, name: str) -> None:
        self.recordings[name].append(time.time())

    def reset(self) -> None:
        self.recordings.clear()

    def get_results(self, name: str, verbose: bool = True) -> Dict[str, Union[Any, Any]]:
        diffs = []
        action_counter = 0
        for i in self.recordings:
            if name in i:
                diff = self.recordings[i][1] - self.recordings[i][0]
                diffs.append(diff)
                action_counter = action_counter + 1
        mean_execution_time = np.round(np.mean(np.array(diffs)), PRECISION)
        total = np.round(action_counter / mean_execution_time, PRECISION)
        if verbose:
            print(
                f"Action '{name}' (repeated {action_counter} times): "
                f"Mean exec-time: {mean_execution_time}, "
                f"Per action (i.e. file): {total} files/s "
                f"Min: {min(diffs)}, Max: {max(diffs)})"
            )
        return {"total": total, "mean": mean_execution_time, "min": min(diffs), "max": max(diffs)}
