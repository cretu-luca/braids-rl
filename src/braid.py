from typing import List

class Braid:
    def __init__(self, word: list[int], n_strands: int):
        self.word = list(word)
        self.n_strands = n_strands
        self.optimal_steps = -1 

    def __len__(self):
        return len(self.word)
    
    def copy(self):
        new_b = Braid(list(self.word), self.n_strands)
        if hasattr(self, 'optimal_steps'):
            new_b.optimal_steps = self.optimal_steps
        return new_b

    def get_padded_word(self, max_len: int) -> List[int]:
        word = self.word[:max_len]
        padding = [0] * (max_len - len(word))
        return word + padding
    
    def check_insert(self, index: int) -> bool:
        return 0 <= index <= len(self.word)
    
    def check_remove_pair(self, index: int) -> bool:
        if index < 0 or index >= len(self.word) - 1:
            return False
        return self.word[index] == -self.word[index + 1]
    
    def check_commutation(self, index: int) -> bool:
        if index < 0 or index >= len(self.word) - 1:
            return False
        
        gen_0, gen_1 = self.word[index], self.word[index + 1]
        return abs(abs(gen_0) - abs(gen_1)) >= 2
    
    def check_braid_relation(self, index: int) -> bool:
        if index < 0 or index >= len(self.word) - 2:
            return False
        
        gen_0, gen_1, gen_2 = self.word[index], self.word[index + 1], self.word[index + 2]
        return gen_0 == gen_2 and \
                abs(abs(gen_0) - abs(gen_1)) == 1 and \
                ((gen_0 > 0 and gen_1 > 0) or (gen_0 < 0 and gen_1 < 0))

    def insert_canceling_pair(self, index: int, generator: int) -> bool:
        if not self.check_insert(index):
            return False
        
        self.word.insert(index, generator)
        self.word.insert(index + 1, -generator)
        return True

    def remove_pair_at_index(self, index: int) -> bool:
        if not self.check_remove_pair(index):
            return False
        
        del self.word[index]
        del self.word[index]
        return True

    def apply_commutation(self, index: int) -> bool:
        if not self.check_commutation(index):
            return False
        
        gen_0 = self.word[index]
        gen_1 = self.word[index + 1]
        
        self.word[index], self.word[index + 1] = gen_1, gen_0
        return True
    
    def apply_braid_relation(self, index: int) -> bool:
        if not self.check_braid_relation(index):
            return False
            
        gen_0 = self.word[index]
        gen_1 = self.word[index+1]
        gen_2 = self.word[index+2]
        
        self.word[index] = gen_1
        self.word[index+1] = gen_0
        self.word[index+2] = gen_1
        return True