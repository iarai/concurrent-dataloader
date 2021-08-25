import statistics
import time
from collections import defaultdict
from functools import wraps
from typing import Any
from typing import Dict
from typing import Union

import numpy as np

PRECISION = 6


class TimeHelper:
    def __init__(self) -> None:
        self.recordings = defaultdict(list)

    def record(self, name: str) -> None:
        print(f"Record started ... {name}")
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
        mean_execution_time = np.round(np.mean(np.array(diffs)), PRECISION) * 1000
        total = np.round(action_counter / mean_execution_time, PRECISION)
        if verbose:
            stdev = 0
            if len(diffs) > 1:
                stdev = statistics.stdev(diffs)
            elif len(diff) == 0:
                return {"total": total, "mean": mean_execution_time, "min": 0, "max": 0}
            print(
                f"Action '{name}' (repeated {action_counter} times): "
                f"Mean exec-time: {mean_execution_time}ms, "
                f"Per action (i.e. file): {total} files/s "
                f"Min: {min(diffs)}, Max: {max(diffs)}) "
                f"std.dv: {stdev}) "
            )
        return {"total": total, "mean": mean_execution_time, "min": min(diffs), "max": max(diffs)}


def stopwatch(trace_name):
    def time_profiler(method):
        @wraps(method)
        def time_profile(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            if "log_time" in kw:
                name = kw.get("log_name", method.__name__.upper())
                kw["log_time"][name] = int((te - ts) * 1000)
            else:
                print(f"({trace_name}) {method.__name__} id ({id(method)}) {(te - ts) * 1000} ms")
            return result

        return time_profile

    return time_profiler
