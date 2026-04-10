# utils/tokenizer.py
# Word-level tokenizer: builds vocabulary from text, encodes/decodes sequences.

import re
from collections import Counter

class Tokenizer:
    """
    Simple word-level tokenizer.
    - Builds a vocabulary from raw text.
    - Encodes text to integer sequences.
    - Decodes integer sequences back to text.
    """

    def __init__(self):
        self.word2idx = {}   # word -> integer index
        self.idx2word = {}   # integer index -> word
        self.vocab_size = 0

        # Special tokens
        self.PAD_TOKEN = "<PAD>"   # padding
        self.UNK_TOKEN = "<UNK>"   # unknown word
        self.SOS_TOKEN = "<SOS>"   # start of sequence
        self.EOS_TOKEN = "<EOS>"   # end of sequence

    def tokenize(self, text):
        """Convert raw text to a list of lowercase word tokens."""
        # Lowercase and split on whitespace/punctuation
        text = text.lower()
        tokens = re.findall(r"[a-zA-Z']+|[.,!?;]", text)
        return tokens

    def build_vocab(self, text):
        """
        Build vocabulary from a string of text.
        Assigns integer indices to each unique word.
        """
        tokens = self.tokenize(text)
        freq = Counter(tokens)

        # Start with special tokens
        special = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        all_words = special + [w for w, _ in freq.most_common()]

        self.word2idx = {w: i for i, w in enumerate(all_words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        print(f"[Tokenizer] Vocabulary size: {self.vocab_size}")
        return self

    def encode(self, text):
        """Convert a string to a list of integer token IDs."""
        tokens = self.tokenize(text)
        unk_id = self.word2idx[self.UNK_TOKEN]
        return [self.word2idx.get(t, unk_id) for t in tokens]

    def decode(self, ids):
        """Convert a list of integer IDs back to a string."""
        words = [self.idx2word.get(i, self.UNK_TOKEN) for i in ids]
        # Filter out special tokens for clean output
        words = [w for w in words if w not in (self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN)]
        return " ".join(words)

    def save(self, path):
        """Save vocabulary to a file."""
        import json
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)
        print(f"[Tokenizer] Saved vocabulary to {path}")

    def load(self, path):
        """Load vocabulary from a file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        self.vocab_size = len(self.word2idx)
        print(f"[Tokenizer] Loaded vocabulary: {self.vocab_size} words")
        return self
