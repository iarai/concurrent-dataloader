from random import Random
from typing import Optional


# TODO use np random generator object
class RandomGenerator:
    def __init__(self, seed: Optional[int] = 42):
        self.rng = Random()
        self.rng.seed(seed)

    def get_int(self, a, b):
        return self.rng.randint(a, b)
