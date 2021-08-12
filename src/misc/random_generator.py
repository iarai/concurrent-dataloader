from random import Random
from typing import Optional

SEED = 10


class RandomGenerator:
    def __init__(self, seed: Optional[int] = SEED):
        self.rng = Random()
        self.rng.seed(SEED)

    def get_int(self, a, b):
        return self.rng.randint(a, b)
