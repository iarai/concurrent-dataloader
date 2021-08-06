import time
from collections import defaultdict

import numpy as np

PRECISION = 6


class TimeHelper:
    def __init__(self):
        self.recordings = defaultdict(list)

    def record(self, name: str):
        self.recordings[name].append(time.time())

    def reset(self):
        self.recordings.clear()

    def get_results(self, name: str, verbose=True):
        diffs = []
        action_counter = 0
        for i in self.recordings:
            if name in i:
                diff = self.recordings[i][1] - self.recordings[i][0]
                diffs.append(diff)
                action_counter = action_counter + 1
        mean_indexing_time = np.round(np.mean(np.array(diffs)), PRECISION)
        total = np.round(action_counter / mean_indexing_time, PRECISION)
        if verbose:
            print(
                f"Action (repeated {action_counter:10}): {name:15} "
                f"Mean action time: {mean_indexing_time:10}, "
                f"Total: {total:10} files/s "
                f"min,max = ({max(diffs), min(diffs)})"
            )
        return {"total": total, "mean": mean_indexing_time, "min": min(diffs), "max": max(diffs)}
