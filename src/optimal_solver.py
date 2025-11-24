import heapq
import time
from typing import List, Tuple, Optional
from .braid import Braid
from .config import Configuration

class AStarSolver:
    def __init__(self, n_strands: int, max_len: int):
        self.n_strands = n_strands
        self.max_len = max_len

    def solve(self, start_braid: Braid, max_time_sec: float = 30.0) -> Optional[List[Tuple[int, int]]]:
        start_time = time.time()
        initial_word = tuple(start_braid.word)
        
        if len(initial_word) == 0:
            return []
        
        start_h = len(initial_word) / 2
        
        queue = [(start_h, 0, initial_word, [])]
        
        visited_cost = {initial_word: 0}
        nodes_expanded = 0

        while queue:
            if time.time() - start_time > max_time_sec:
                return None # Timeout

            _, cost, current_word_tuple, history = heapq.heappop(queue)
            nodes_expanded += 1

            if current_word_tuple in visited_cost and visited_cost[current_word_tuple] < cost:
                continue

            if len(current_word_tuple) == 0:
                return history

            current_braid = Braid(list(current_word_tuple), self.n_strands)
            curr_len = len(current_braid)

            potential_moves = []
            
            for i in range(curr_len - 1):
                if current_braid.check_commutation(i):
                    new_word = list(current_word_tuple)
                    new_word[i], new_word[i+1] = new_word[i+1], new_word[i]
                    potential_moves.append((0, i, tuple(new_word)))

            for i in range(curr_len - 2):
                if current_braid.check_braid_relation(i):
                    new_word = list(current_word_tuple)
                    new_word[i], new_word[i+1], new_word[i+2] = new_word[i+1], new_word[i], new_word[i+1]
                    potential_moves.append((1, i, tuple(new_word)))

            for i in range(curr_len - 1):
                if current_braid.check_remove_pair(i):
                    new_word = list(current_word_tuple)
                    del new_word[i]
                    del new_word[i]
                    potential_moves.append((2, i, tuple(new_word)))

            if curr_len < self.max_len - 2 and curr_len < len(initial_word) + 6:
                for i in range(curr_len + 1):
                    for gen in range(1, self.n_strands):
                        new_word = list(current_word_tuple)
                        new_word.insert(i, gen)
                        new_word.insert(i+1, -gen)
                        potential_moves.append((3, i, tuple(new_word)))

            for move_type, index, new_word_tuple in potential_moves:
                new_cost = cost + 1
                
                if new_word_tuple not in visited_cost or new_cost < visited_cost[new_word_tuple]:
                    visited_cost[new_word_tuple] = new_cost
                    priority = new_cost + (len(new_word_tuple) / 2)
                    heapq.heappush(queue, (priority, new_cost, new_word_tuple, history + [(move_type, index)]))

        return None