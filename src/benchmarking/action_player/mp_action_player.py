import logging
from functools import partial
from multiprocessing import Pool
from multiprocessing import set_start_method
from pathlib import Path
from typing import Callable

from benchmarking.action_player.action_player import ActionPlayer
from benchmarking.misc.logging_configuration import initialize_logging

try:
    set_start_method("spawn")
except RuntimeError:
    pass


class MPActionPlayer(ActionPlayer):
    def __init__(self, pool_size: int = 2) -> None:
        super().__init__()
        logging.debug("Initializing Multiprocessing ActionPlayer")
        self.pool_size = pool_size

    def run(self, action_name: str, action: Callable):
        action()

    def benchmark(self, action_name: str, action: Callable, repeat: int, output_base_folder: Path,) -> None:
        # each worker is assigned a number of repetitions (so in total still "repeat" number of actions)
        with Pool(
            self.pool_size,
            initializer=partial(
                initialize_logging,
                loglevel=logging.getLogger().getEffectiveLevel(),
                output_base_folder=output_base_folder,
            ),
        ) as pool:
            pool.starmap(self.run, [(action_name, action)] * self.pool_size)
