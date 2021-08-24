from random import Random
from typing import Optional

SEED = 10


class RandomGenerator:
    def __init__(self, seed: Optional[int] = SEED):
        self.rng = Random()
        self.rng.seed(SEED)
        self.range = []
        self.current = 0
        self._generate_range()

    def get_next_int(self) -> int:
        self.current += 1
        if self.current > len(self.range) - 1:
            self._generate_range()
        return self.range[self.current]

    def _generate_range(self) -> int:
        self.range = [self.rng.randrange(0, 1000, 1) for i in range(1000)]

    def get_int(self, a, b):
        return self.rng.randint(a, b)
