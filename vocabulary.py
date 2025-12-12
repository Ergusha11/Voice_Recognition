class Vocabulary:
    def __init__(self):
        # Define valid characters: lowercase spanish alphabet + space
        self.chars = " abcdefghijklmnñopqrstuvwxyzáéíóú"
        # Map char to index
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        # Map index to char
        self.idx2char = {idx + 1: char for idx, char in enumerate(self.chars)}
        # 0 is reserved for the CTC "Blank" token
        self.BLANK_IDX = 0

    def text_to_int(self, text):
        """Converts a string to a list of integers."""
        text = text.lower().strip()
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def int_to_text(self, sequence):
        """
        Decodes a sequence of integers back to text (Greedy Decoder).
        Collapses repeated characters and removes blanks.
        """
        decoded = []
        last_idx = -1
        
        for idx in sequence:
            if idx == self.BLANK_IDX:
                last_idx = -1 # Reset
                continue
            
            if idx != last_idx:
                decoded.append(self.idx2char[idx])
            
            last_idx = idx
            
        return "".join(decoded)

    def __len__(self):
        return len(self.chars) + 1 # +1 for blank token