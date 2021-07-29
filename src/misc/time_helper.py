import time
from collections import defaultdict
import numpy as np

PRECISION = 6

class TimeHelper:
    def __init__(
        self,
    ):
        self.recordings = defaultdict(list)

    def record(self, name: str):
        self.recordings[name].append(time.time())

    def reset(self):
        self.recordings.clear()

    def get_results(self, name: str, summary_only=True):
        diffs = []
        action_counter = 0
        for i in self.recordings:
            if name in i:
                diff = self.recordings[i][1] - self.recordings[i][0]
                diffs.append(diff)
                action_counter = action_counter + 1
                if not summary_only:
                    print(f"{i} current: {diff} s mean: {action_counter/diff} files/s")
        mean_indexing_time = np.round(np.mean(np.array(diffs)), PRECISION)
        print(
            f"Action: {name:15} "
            f"Mean action time: {mean_indexing_time:15}, "
            f"Total: {np.round(action_counter/mean_indexing_time, PRECISION):10} files/s"
        )
