import logging
from typing import Callable


class ActionPlayer:
    def __init__(self) -> None:
        logging.debug("Initializing ActionPlayer")

    def benchmark(self, action_name: str, action: Callable, repeat: int, **kwargs) -> None:
        logging.info(f"Repeating {action_name} {repeat} times!")
        for i in range(repeat):
            logging.debug(f"Benchmark starts: {i}")
            action()
            logging.debug(f"Benchmark done: {i}")
        logging.info(f"Done epeating {action_name} {repeat} times!")
