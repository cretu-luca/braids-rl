from braid import Braid
import random

class BraidEngine:
    def __init__(self, n_strands: int, max_length: int):
        self.n_strands = n_strands
        self.max_length = max_length

    def generate(self, complexity: int):
        word = []

        for _ in range(complexity):
            gen = random.randint(1, self.n_strands - 1)

            if len(word) == 0:
                insert_pos = 0
            else:
                insert_pos = random.randint(0, len(word))

            word.insert(insert_pos, gen)
            word.insert(insert_pos + 1, gen)


        return Braid(word, self.n_strands)