from typing import Callable

from misc.time_helper import TimeHelper


class ActionPlayer:
    def __init__(self) -> None:
        self.stopwatch = TimeHelper()

    def reset(self) -> None:
        self.stopwatch.reset()

    def benchmark(self, action_name: str, action: Callable, repeat: int) -> None:
        for i in range(repeat):
            self.stopwatch.record(action_name + "_" + str(i))
            action()
            self.stopwatch.record(action_name + "_" + str(i))
        self.stopwatch.get_results(action_name)
