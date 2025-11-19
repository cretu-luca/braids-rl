class Braid:
    def __init__(self, word: list[int], n_strands: int):
        self.word = list(word)
        self.n_strands = n_strands

    def __repr__(self):
        return f"Braid(n={self.n_strands}, word={self.word})"

    def __len__(self):
        return len(self.word)

    def reduce(self):
        stack = []
        changed = False

        for generator in self.word:
            if stack and stack[-1] == -generator:
                stack.pop()
                changed = True
            else:
                stack.append(generator)

        self.word = stack
        return changed

    def get_commutation_moves(self):
        indices = []

        for i in range(len(self.word) - 1):
            a = abs(self.word[i])
            b = abs(self.word[i+1])

            if abs(a - b) >= 2:
                indices.append(i)

        return indices

    def apply_commutation(self, index):
        valid_indices = self.get_commutation_moves()
        if index not in valid_indices:
            return False

        self.word[index], self.word[index+1] = self.word[index+1], self.word[index]
        return True

    def get_braid_relation_moves(self):
        indices = []
        for i in range(len(self.word) - 2):
            a = self.word[i]
            b = self.word[i+1]
            c = self.word[i+2]
            
            if a == c:
                if abs(abs(a) - abs(b)) == 1:
                    indices.append(i)
        return indices

    def apply_braid_relation(self, index):
        valid_indices = self.get_braid_relation_moves()
        if index not in valid_indices:
            return False
            
        a = self.word[index]
        b = self.word[index+1]
        
        self.word[index] = b
        self.word[index+1] = a
        self.word[index+2] = b
        return True