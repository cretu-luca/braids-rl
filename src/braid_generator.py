import ast
import random
from typing import List, Optional
from sage.all import BraidGroup

from .braid import Braid
from .config import Configuration

class BraidGenerator:
    def __init__(self, n_strands: int, config: Configuration):
        self.n_strands = n_strands
        self.config = config
        self.B = BraidGroup(n_strands)

    def generate_braid(self, crossings: int, difficulty: int) -> Braid:
        braid = Braid([], self.n_strands)

        while len(braid) < crossings:
            generator = random.randint(1, self.n_strands - 1)
            index = random.randint(0, len(braid))

            braid.insert_canceling_pair(index, generator)
        
        moves = 0
        attempts = 0
        max_attempts = 1000

        while moves < difficulty and attempts < max_attempts:
            possible_moves = []
            attempts += 1

            for i in range(len(braid) - 2):
                gen_0, gen_1, gen_2 = braid.word[i], braid.word[i + 1], braid.word[i + 2]

                if gen_0 == gen_2 and abs(abs(gen_0) - abs(gen_1)) == 1:
                    if gen_0 * gen_1 > 0: # same sign
                        possible_moves.append(('r3', i))

            for i in range(len(braid) - 1):
                if abs(abs(braid.word[i]) - abs(braid.word[i+1])) >= 2:
                    possible_moves.append(('commute', i))

            if possible_moves: 
                move, index = random.choice(possible_moves)

                if move == 'r3':
                    braid.apply_braid_relation(index)
                else:
                    braid.apply_commutation(index)

                moves += 1

        return braid

    def generate_dataset(self, count: int, crossings: int, difficulty: int, filepath: Optional[str] = None):
        print(f"generating dataset of {count} braids with {crossings} crossings of difficulty {difficulty}")

        if not filepath: 
            filepath = f"{self.config.DATA_DIR}braids_{self.n_strands}st_{crossings}cr_{difficulty}dif"

        with open(filepath, 'w') as file:
            file.write(f"{count},{self.n_strands},{crossings},{difficulty}\n")

            generated = 0
            while generated < count:
                word = self.generate_braid(crossings, difficulty).word
                braid = self.B(word)

                if braid.is_one():
                    generated += 1
                    file.write(f"{word}\n")

        print("done")

    @staticmethod
    def load_dataset(filepath: str) -> List[Braid]:
        braids = []

        with open(filepath, 'r') as file:
            header = file.readline().strip()

            if header:
                _, n_strands, _, _ = map(int, header.split(','))

            for line in file:
                try:
                    word = ast.literal_eval(line.strip())
                    if isinstance(word, list):
                        braids.append(Braid(word, n_strands))
                except Exception as _:
                    continue

        return braids