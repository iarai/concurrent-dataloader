import random
from multiprocessing import Pool
from multiprocessing import set_start_method
from typing import Callable

from misc.action_player import ActionPlayer

try:
    set_start_method("spawn")
except RuntimeError:
    pass


class MPActionPlayer(ActionPlayer):
    def __init__(self, num_workers: int = 4, pool_size: int = 2) -> None:
        super().__init__()
        print("Initializing Multiprocessing ActionPlayer")
        self.num_workers = num_workers
        self.pool_size = pool_size

    def run(self, action_name: str, action: Callable, repeat_action: int = 20):
        pool_id = random.randint(0, 1000)
        action_name = action_name + "_pooled_run_" + action.__name__ + "_" + str(pool_id)
        for i in range(repeat_action):
            self.stopwatch.record(action_name + str(i))
            action()
            self.stopwatch.record(action_name + str(i))
        return self.stopwatch.get_results(action_name, False)["total"]

    def benchmark(self, action_name: str, action: Callable, repeat: int) -> None:
        with Pool(self.pool_size) as pool:
            results = pool.starmap(self.run, [(action_name, action, repeat // self.num_workers)] * self.num_workers)
        print(f"Results: {results}, sum = {sum(results)}")
