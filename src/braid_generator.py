import ast
import random
import os
from typing import List, Optional

try:
    from sage.all import BraidGroup
except ImportError:
    print("invalid sage import")

from .braid import Braid
from .config import Configuration
from .optimal_solver import AStarSolver

class BraidGenerator:
    def __init__(self, n_strands: int, config: Configuration, seed: Optional[int] = None):
        self.n_strands = n_strands
        self.config = config
        self.B = BraidGroup(n_strands)

        self.rng = random.Random(seed)

        self.solver = AStarSolver(n_strands, config.MAX_LEN)

    def generate_braid(self, crossings: int, difficulty: int) -> Braid:
        valid = False

        while not valid:
            braid = Braid([], self.n_strands)

            while len(braid) < crossings:
                generator = self.rng.randint(1, self.n_strands - 1)
                index = self.rng.randint(0, len(braid))

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
                        if gen_0 * gen_1 > 0:
                            possible_moves.append(('r3', i))

                for i in range(len(braid) - 1):
                    if abs(abs(braid.word[i]) - abs(braid.word[i+1])) >= 2:
                        possible_moves.append(('commute', i))

                if possible_moves: 
                    move, index = self.rng.choice(possible_moves)

                    if move == 'r3':
                        braid.apply_braid_relation(index)
                    else:
                        braid.apply_commutation(index)

                    moves += 1

            sage_braid = self.B(braid.word)

            if sage_braid.is_one():
                valid = True

        return braid

    def generate_dataset(self, count: int, crossings: int, difficulty: int, filepath: Optional[str] = None, compute_optimal: bool = False):
        print(f"Generating dataset: {count} braids, {crossings} crossings (Optimal={compute_optimal})...")
        
        if not filepath:
            os.makedirs(self.config.DATA_DIR, exist_ok=True)
            filepath = os.path.join(self.config.DATA_DIR, f"braids_{self.n_strands}st_{crossings}cr_opt.txt")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as file:
            file.write(f"{count},{self.n_strands},{crossings},{difficulty},optimal={compute_optimal}\n")

            generated_count = 0
            while generated_count < count:
                braid_obj = self.generate_braid(crossings, difficulty)
                
                line_content = str(braid_obj.word)
                
                if compute_optimal:
                    path = self.solver.solve(braid_obj, max_time_sec=10.0)
                    optimal_steps = len(path) if path is not None else -1

                    line_content = f"{line_content}, {optimal_steps}"

                file.write(f"{line_content}\n")
                generated_count += 1

        print(f"Done. Saved to {filepath}")

    @staticmethod
    def load_dataset(filepath: str) -> List[Braid]:
        braids = []
        if not os.path.exists(filepath):
            return []

        with open(filepath, 'r') as file:
            header = file.readline().strip()
            n_strands = 3
            if header:
                try:
                    parts = header.split(',')
                    if len(parts) >= 2: n_strands = int(parts[1])
                except: pass

            for line in file:
                try:
                    parsed = ast.literal_eval(line.strip())
                    word = []
                    opt_steps = None
                    
                    if isinstance(parsed, tuple):
                        word = parsed[0]
                        opt_steps = parsed[1]
                    elif isinstance(parsed, list):
                        word = parsed
                    
                    if isinstance(word, list):
                        b = Braid(word, n_strands)
                        if opt_steps is not None:
                            b.optimal_steps = opt_steps
                        braids.append(b)
                except:
                    continue
        return braids