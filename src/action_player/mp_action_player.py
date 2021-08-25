import logging
from multiprocessing import Pool
from multiprocessing import set_start_method
from typing import Callable

from action_player.action_player import ActionPlayer

try:
    set_start_method("spawn")
except RuntimeError:
    pass


class MPActionPlayer(ActionPlayer):
    def __init__(self, rng, num_workers: int = 4, pool_size: int = 2) -> None:
        super().__init__()
        # assert rng is None, "Please make an external random number generator, here we use pooling"
        logging.debug("Initializing Multiprocessing ActionPlayer")
        self.num_workers = num_workers
        self.pool_size = pool_size
        self.rng = rng

    def run(self, action_name: str, action: Callable, repeat_action: int = 20):
        pool_id = self.rng.get_int(0, 1000)  # npr.randint(0, 1000)
        logging.debug("Repeating {action_name} {repeat_action} times!")
        action_name = action_name + "_pooled_run_" + action.__name__ + "_" + str(pool_id)
        print(f"Repeating... {repeat_action}")
        for _ in range(repeat_action):
            self.stopwatch.record(action_name)
            action()
            self.stopwatch.record(action_name)
        return self.stopwatch.get_results(action_name, False)["total"]

    def benchmark(self, action_name: str, action: Callable, repeat: int, verbose: bool = False) -> None:
        # each worker is assigned a number of repetitions (so in total still "repeat" number of actions)
        assert repeat >= self.num_workers, "Number of repetitions needs to be higher than number of workers..."
        with Pool(self.pool_size) as pool:
            results = pool.starmap(self.run, [(action_name, action, repeat // self.num_workers)] * self.num_workers)
        if verbose:
            logging.info(f"Results (time per paralel process): {results}, sum = {sum(results)}")
