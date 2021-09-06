import logging
from typing import Callable


class ActionPlayer:
    def __init__(self) -> None:
        logging.debug("Initializing ActionPlayer")

    def reset(self) -> None:
        self.stopwatch.reset()

    def benchmark(self, action_name: str, action: Callable, repeat: int, **kwargs) -> None:
        for i in range(repeat):
            logging.info(f"Benchmark starts: {i}")
            action()
            logging.info(f"Benchmark done: {i}")
