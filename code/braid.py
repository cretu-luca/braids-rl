class Braid:
    def __init__(self, word: list[int], n_strands: int):
        self.word = list(word)
        self.n_strands = n_strands

    def __repr__(self):
        return f"Braid(n={self.n_strands}, word={self.word})"

    def __len__(self):
        return len(self.word)
    
    def copy(self):
        return Braid(list(self.word), self.n_strands)

    def reduce(self):
        stack = []
        changed = False

        for generator in self.word:
            if stack and stack[-1] == -generator:
                stack.pop()
                changed = True
            else:
                stack.append(generator)

        original_len = len(self.word)
        self.word = stack
        return len(self.word) < original_len

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
        
        if a == c:
            if abs(abs(a) - abs(b)) == 1:
                self.word[index] = b
                self.word[index+1] = a
                self.word[index+2] = b
                return True
            
        return False