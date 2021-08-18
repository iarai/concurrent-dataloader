import logging
from typing import Callable

from misc.time_helper import TimeHelper


class ActionPlayer:
    def __init__(self) -> None:
        logging.debug("Initializing ActionPlayer")
        self.stopwatch = TimeHelper()

    def reset(self) -> None:
        self.stopwatch.reset()

    def benchmark(self, action_name: str, action: Callable, repeat: int, verbose: bool = False) -> None:
        for i in range(repeat):
            print(f"Benchmark starts: {i}")
            self.stopwatch.record(action_name + "_" + str(i))
            action()
            self.stopwatch.record(action_name + "_" + str(i))
            print(f"Benchmark done: {i}")
        self.stopwatch.get_results(action_name, verbose)
