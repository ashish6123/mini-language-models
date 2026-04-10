# models/transformer.py
# Mini GPT-style Transformer Language Model — built entirely from scratch.
# No HuggingFace. No external transformer libraries.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 1. POSITIONAL ENCODING
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings using sine/cosine functions.

    Without this, the transformer has no notion of word order.
    The encoding at position `pos` and dimension `i` is:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encoding matrix once
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos

        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)  # Not a parameter, but saved with model

    def forward(self, x):
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 2. MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (causal / masked).

    Each 'head' learns to attend to different parts of the sequence.
    'Causal' masking ensures position i can only attend to positions <= i,
    which is required for autoregressive (left-to-right) language modeling.

    Steps:
        1. Project input into Q, K, V matrices.
        2. Compute attention scores: softmax(QK^T / sqrt(d_k)).
        3. Apply causal mask (upper triangle = -inf).
        4. Weighted sum of V.
        5. Concatenate heads and project back.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Optional causal mask of shape (seq_len, seq_len).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, T, C = x.shape  # batch, sequence length, d_model

        # Step 1: Compute Q, K, V
        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Reshape for multi-head: (B, num_heads, T, d_k)
        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Step 3: Scaled dot-product attention
        # scores: (B, num_heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Step 4: Apply causal mask (prevent attending to future tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Step 5: Softmax over last dimension (over positions)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Step 6: Weighted sum of V
        context = torch.matmul(attn_weights, V)  # (B, num_heads, T, d_k)

        # Step 7: Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(context)


# ─────────────────────────────────────────────
# 3. FEED-FORWARD NETWORK
# ─────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).

    Applied independently to each position after attention.
    Expands dimensionality by 4x, applies ReLU, then projects back.
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model  # Default inner dim is 4x d_model

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 4. TRANSFORMER BLOCK
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block:
        - Multi-head self-attention (with causal mask)
        - Add & Norm (residual connection + layer normalization)
        - Feed-forward network
        - Add & Norm

    Residual connections help gradients flow during backprop.
    Layer norm stabilizes activations.
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention block with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # Feed-forward block with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x


# ─────────────────────────────────────────────
# 5. MINI GPT TRANSFORMER
# ─────────────────────────────────────────────

class MiniTransformer(nn.Module):
    """
    Mini GPT-style Transformer Language Model.

    Full architecture:
        Token Embedding  +  Positional Encoding
              ↓
        N × Transformer Block (attention + FFN)
              ↓
        Layer Norm
              ↓
        Linear (vocab projection)
              ↓
        Logits over vocabulary
    """

    def __init__(
        self,
        vocab_size,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=128,
        dropout=0.1,
    ):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Embedding and model dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer blocks.
            max_seq_len (int): Maximum input sequence length.
            dropout (float): Dropout probability.
        """
        super().__init__()

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm before output
        self.norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Initialize weights (improves training stability)
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, seq_len, device):
        """
        Create a causal (lower-triangular) mask.
        Position i can only attend to positions 0..i.
        Shape: (seq_len, seq_len), 1 = allow, 0 = block.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T) for broadcasting

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Token IDs, shape (batch, seq_len).

        Returns:
            logits (Tensor): Shape (batch, seq_len, vocab_size).
        """
        B, T = x.shape
        device = x.device

        # Embed tokens and add positional encoding
        tok_emb = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(tok_emb)

        # Create causal mask
        mask = self._causal_mask(T, device)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization
        x = self.norm(x)

        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    def generate(self, seed_ids, tokenizer, num_words=30, temperature=1.0, device="cpu"):
        """
        Autoregressive text generation.

        Generates one token at a time:
        → feed all tokens so far → get logits for last position → sample → append → repeat.

        Args:
            seed_ids (list[int]): Starting token IDs.
            tokenizer: Tokenizer object for decoding.
            num_words (int): Number of tokens to generate.
            temperature (float): Sampling temperature.
            device (str): 'cpu' or 'cuda'.

        Returns:
            str: Generated text.
        """
        self.eval()
        generated_ids = list(seed_ids)

        with torch.no_grad():
            for _ in range(num_words):
                # Truncate to max_seq_len if needed
                context = generated_ids[-self.max_seq_len:]
                x = torch.tensor([context], dtype=torch.long).to(device)

                # Forward pass
                logits = self.forward(x)  # (1, seq_len, vocab_size)

                # Get logits for the LAST position (next token prediction)
                last_logits = logits[:, -1, :] / max(temperature, 1e-8)

                # Sample from distribution
                probs = F.softmax(last_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                generated_ids.append(next_id)

        return tokenizer.decode(generated_ids)
