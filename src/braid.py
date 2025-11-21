from typing import List, Optional

class Braid:
    def __init__(self, word: list[int], n_strands: int):
        self.word = list(word)
        self.n_strands = n_strands

    def __len__(self):
        return len(self.word)
    
    def copy(self):
        return Braid(list(self.word), self.n_strands)

    def get_padded_word(self, max_len: int) -> List[int]:
        w = self.word[:max_len]
        padding = [0] * (max_len - len(w))
        return w + padding

    def insert_canceling_pair(self, index: int, generator: int) -> bool:
        if index < 0 or index > len(self.word):
            return False
        
        self.word.insert(index, generator)
        self.word.insert(index + 1, -generator)
        return True

    def remove_pair_at_index(self, index: int):
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
            
        a = self.word[index]
        b = self.word[index+1]
        
        if abs(abs(a) - abs(b)) >= 2:
            self.word[index], self.word[index+1] = b, a
            return True
        return False

    def apply_braid_relation(self, index):
        if index < 0 or index >= len(self.word) - 2:
            return False
            
        a = self.word[index]
        b = self.word[index+1]
        c = self.word[index+2]
        
        if a == c and abs(abs(a) - abs(b)) == 1:
            if (a > 0 and b > 0) or (a < 0 and b < 0):
                self.word[index] = b
                self.word[index+1] = a
                self.word[index+2] = b
                return True
            
        return False