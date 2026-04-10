# models/ngram.py
# N-gram language model using probability tables.
# Supports bigram (n=2) and trigram (n=3) models.

import random
from collections import defaultdict, Counter


class NGramModel:
    """
    N-gram Language Model.

    Counts how often each sequence of (n-1) words is followed by each word,
    then uses those counts to compute probabilities for the next word.

    Example (bigram):
        "the cat sat" -> P("sat" | "cat") = count("cat sat") / count("cat ...")
    """

    def __init__(self, n=2):
        """
        Args:
            n (int): Order of the n-gram. 2 = bigram, 3 = trigram.
        """
        assert n >= 2, "N must be at least 2 (bigram)"
        self.n = n
        # counts[(w1, w2, ...)] -> Counter({next_word: count})
        self.counts = defaultdict(Counter)
        # Total occurrences of each context (for normalization)
        self.context_totals = defaultdict(int)
        self.vocab = set()

    def train(self, token_list):
        """
        Train the model by counting n-gram occurrences.

        Args:
            token_list (list[str]): List of word tokens.
        """
        self.vocab = set(token_list)
        # Slide a window of size n over the token list
        for i in range(len(token_list) - self.n + 1):
            gram = token_list[i : i + self.n]
            context = tuple(gram[:-1])    # (n-1) words as context
            next_word = gram[-1]           # the word to predict

            self.counts[context][next_word] += 1
            self.context_totals[context] += 1

        print(f"[NGram] Trained {self.n}-gram model | {len(self.counts)} contexts | vocab={len(self.vocab)}")

    def probability(self, context, word):
        """
        Compute P(word | context) with simple Laplace (add-1) smoothing.

        Args:
            context (tuple[str]): The (n-1) preceding words.
            word (str): The candidate next word.

        Returns:
            float: Probability of 'word' following 'context'.
        """
        count_context_word = self.counts[context].get(word, 0)
        count_context = self.context_totals.get(context, 0)
        # Laplace smoothing: add 1 to numerator, vocab_size to denominator
        return (count_context_word + 1) / (count_context + len(self.vocab) + 1)

    def predict_next(self, context, top_k=5, temperature=1.0):
        """
        Sample the next word given a context.

        Args:
            context (tuple[str]): Previous (n-1) words.
            top_k (int): Consider only top-k candidates.
            temperature (float): Sampling temperature (higher = more random).

        Returns:
            str: Predicted next word.
        """
        # If context is unseen, fall back to random vocab word
        if context not in self.counts:
            return random.choice(list(self.vocab)) if self.vocab else "<UNK>"

        # Get all candidate next words and their counts
        candidates = list(self.counts[context].items())
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:max(top_k, len(candidates))]

        words, counts = zip(*top_candidates)
        # Apply temperature scaling
        import math
        scores = [c ** (1.0 / max(temperature, 1e-8)) for c in counts]
        total = sum(scores)
        probs = [s / total for s in scores]

        # Sample from the probability distribution
        return random.choices(words, weights=probs, k=1)[0]

    def generate(self, seed_tokens, num_words=30, temperature=1.0):
        """
        Generate text starting from a seed sequence.

        Args:
            seed_tokens (list[str]): Starting words.
            num_words (int): Number of words to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: Generated text.
        """
        # Pad seed if shorter than context window
        context_len = self.n - 1
        tokens = list(seed_tokens)
        if len(tokens) < context_len:
            tokens = ["the"] * (context_len - len(tokens)) + tokens

        generated = list(tokens)

        for _ in range(num_words):
            context = tuple(generated[-context_len:])
            next_word = self.predict_next(context, temperature=temperature)
            generated.append(next_word)

        # Return only the newly generated part
        return " ".join(generated)

    def save(self, path):
        """Save model to disk using pickle."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "n": self.n,
                "counts": dict(self.counts),
                "context_totals": dict(self.context_totals),
                "vocab": self.vocab
            }, f)
        print(f"[NGram] Model saved to {path}")

    def load(self, path):
        """Load model from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.n = data["n"]
        self.counts = defaultdict(Counter, data["counts"])
        self.context_totals = defaultdict(int, data["context_totals"])
        self.vocab = data["vocab"]
        print(f"[NGram] Model loaded from {path}")
        return self
