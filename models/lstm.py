# models/lstm.py
# LSTM-based language model for next-token prediction.
# Architecture: Embedding -> LSTM -> Linear (vocab projection)

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMLanguageModel(nn.Module):
    """
    LSTM Language Model.

    Architecture:
        1. Embedding layer: maps token IDs to dense vectors.
        2. LSTM layer: processes sequences, captures temporal dependencies.
        3. Linear layer: projects LSTM output to vocabulary logits.

    Training objective: predict the next token at each position (causal LM).
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        """
        Args:
            vocab_size (int): Number of unique tokens in vocabulary.
            embed_dim (int): Dimensionality of token embeddings.
            hidden_dim (int): Number of hidden units in LSTM.
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout probability between LSTM layers.
        """
        super(LSTMLanguageModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # --- Layers ---
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM: processes embedded tokens sequentially
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True   # input shape: (batch, seq, features)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Output projection: hidden_dim -> vocab_size (logits)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass.

        Args:
            x (Tensor): Input token IDs, shape (batch, seq_len).
            hidden: Previous LSTM hidden state (h_n, c_n) or None.

        Returns:
            logits (Tensor): Shape (batch, seq_len, vocab_size).
            hidden: Updated LSTM hidden state.
        """
        # Step 1: Embed input tokens
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)

        # Step 2: Pass through LSTM
        output, hidden = self.lstm(embedded, hidden)  # output: (batch, seq_len, hidden_dim)

        # Step 3: Apply dropout to LSTM output
        output = self.dropout(output)

        # Step 4: Project to vocabulary size
        logits = self.fc(output)  # (batch, seq_len, vocab_size)

        return logits, hidden

    def init_hidden(self, batch_size, device):
        """Initialize hidden state to zeros."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def generate(self, seed_ids, tokenizer, num_words=30, temperature=1.0, device="cpu"):
        """
        Generate text autoregressively from a seed sequence.

        Args:
            seed_ids (list[int]): Encoded seed token IDs.
            tokenizer: Tokenizer object (for decoding).
            num_words (int): Number of new tokens to generate.
            temperature (float): Controls randomness (lower = more deterministic).
            device (str): 'cpu' or 'cuda'.

        Returns:
            str: Generated text string.
        """
        self.eval()
        generated_ids = list(seed_ids)

        with torch.no_grad():
            hidden = self.init_hidden(1, device)

            for _ in range(num_words):
                # Prepare input tensor from last generated token
                x = torch.tensor([[generated_ids[-1]]], dtype=torch.long).to(device)

                # Forward pass
                logits, hidden = self.forward(x, hidden)  # logits: (1, 1, vocab_size)

                # Apply temperature scaling then sample
                logits = logits[:, -1, :] / max(temperature, 1e-8)  # (1, vocab_size)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(next_id)

        return tokenizer.decode(generated_ids)
