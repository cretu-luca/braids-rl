from typing import List

class Braid:
    def __init__(self, word: list[int], n_strands: int):
        self.word = list(word)
        self.n_strands = n_strands

    def __len__(self):
        return len(self.word)
    
    def copy(self):
        return Braid(list(self.word), self.n_strands)

    def get_padded_word(self, max_len: int) -> List[int]:
        word = self.word[:max_len]
        padding = [0] * (max_len - len(word))
        return word + padding

    def insert_canceling_pair(self, index: int, generator: int) -> bool:
        if index < 0 or index > len(self.word):
            return False
        
        self.word.insert(index, generator)
        self.word.insert(index + 1, -generator)
        return True

    def remove_pair_at_index(self, index: int) -> bool:
        if index < 0 or index >= len(self.word) - 1:
            return False
        
        if self.word[index] == -self.word[index + 1]:
            del self.word[index]
            del self.word[index]
            return True
        return False

    def reduce_global(self):
        stack = []
        for generator in self.word:
            if stack and stack[-1] == -generator:
                stack.pop()
            else:
                stack.append(generator)
        
        changed = len(stack) < len(self.word)
        self.word = stack
        return changed

    def apply_commutation(self, index):
        if index < 0 or index >= len(self.word) - 1:
            return False
            
        gen_0 = self.word[index]
        gen_1 = self.word[index + 1]
        
        if abs(abs(gen_0) - abs(gen_1)) >= 2:
            self.word[index], self.word[index + 1] = gen_1, gen_0
            return True
        return False

    def apply_braid_relation(self, index):
        if index < 0 or index >= len(self.word) - 2:
            return False
            
        gen_0 = self.word[index]
        gen_1 = self.word[index+1]
        gen_2 = self.word[index+2]
        
        if gen_0 == gen_2 and abs(abs(gen_0) - abs(gen_1)) == 1:
            if (gen_0 > 0 and gen_1 > 0) or (gen_0 < 0 and gen_1 < 0):
                self.word[index] = gen_1
                self.word[index+1] = gen_0
                self.word[index+2] = gen_1
                return True
            
        return False