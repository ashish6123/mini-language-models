# train.py
# Master training script — trains all 3 models and saves them to disk.
# Run: python train.py

import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# ── Local imports ──────────────────────────────────────────
from utils.tokenizer import Tokenizer
from utils.dataset import get_dataloader
from models.ngram import NGramModel
from models.lstm import LSTMLanguageModel
from models.transformer import MiniTransformer

# ── Config ─────────────────────────────────────────────────
DATA_PATH      = "data/sample.txt"
SAVE_DIR       = "saved_models"
VOCAB_PATH     = os.path.join(SAVE_DIR, "vocab.json")

# Training hyperparameters (kept small for fast training)
SEQ_LEN        = 32
BATCH_SIZE     = 16
LSTM_EPOCHS    = 20
TRANS_EPOCHS   = 20
LEARNING_RATE  = 3e-3

# Model hyperparameters
EMBED_DIM      = 64
HIDDEN_DIM     = 128
NUM_LAYERS     = 2
D_MODEL        = 64
NUM_HEADS      = 4
TRANS_LAYERS   = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────────────────────────────────────────
# UTILITIES
# ───────────────────────────────────────────────────────────

def separator(title):
    print("\n" + "="*55)
    print(f"  {title}")
    print("="*55)


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def train_epoch(model, dataloader, criterion, optimizer):
    """Run one epoch of training. Returns average loss."""
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(x) if hasattr(model, 'init_hidden') else (model(x), None)

        # Reshape for cross-entropy: (batch*seq, vocab)
        B, T, V = logits.shape
        loss = criterion(logits.view(B * T, V), y.view(B * T))

        loss.backward()
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ───────────────────────────────────────────────────────────
# STEP 1 — TOKENIZER & DATA
# ───────────────────────────────────────────────────────────

def prepare_data():
    separator("STEP 1: Preparing Data & Tokenizer")

    text = load_text(DATA_PATH)
    print(f"Loaded text: {len(text)} characters")

    # Build vocabulary
    tokenizer = Tokenizer()
    tokenizer.build_vocab(text)

    # Save vocabulary for later use (generate.py, app.py)
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save(VOCAB_PATH)

    # Encode entire text as token IDs
    token_ids = tokenizer.encode(text)
    print(f"Encoded into {len(token_ids)} tokens")

    # Also get raw tokens (words) for n-gram training
    token_words = tokenizer.tokenize(text)

    return tokenizer, token_ids, token_words


# ───────────────────────────────────────────────────────────
# STEP 2 — N-GRAM MODEL
# ───────────────────────────────────────────────────────────

def train_ngram(token_words):
    separator("STEP 2: Training N-Gram Model (Trigram)")

    ngram = NGramModel(n=3)
    ngram.train(token_words)

    save_path = os.path.join(SAVE_DIR, "ngram.pkl")
    ngram.save(save_path)

    # Quick sanity check
    seed = token_words[:2]
    generated = ngram.generate(seed, num_words=15)
    print(f"\n[NGram Sample] {generated}")

    return ngram


# ───────────────────────────────────────────────────────────
# STEP 3 — LSTM MODEL
# ───────────────────────────────────────────────────────────

def train_lstm(tokenizer, token_ids):
    separator("STEP 3: Training LSTM Language Model")

    # Build data loader
    dataloader = get_dataloader(token_ids, SEQ_LEN, BATCH_SIZE)

    # Initialize model
    model = LSTMLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.3
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD token
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    print(f"\nTraining for {LSTM_EPOCHS} epochs on {DEVICE}...")

    best_loss = float("inf")
    for epoch in range(1, LSTM_EPOCHS + 1):
        start = time.time()

        # Custom training loop that handles LSTM hidden state
        model.train()
        total_loss = 0
        criterion_fn = nn.CrossEntropyLoss(ignore_index=0)

        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            logits, _ = model(x)
            B, T, V = logits.shape
            loss = criterion_fn(logits.view(B * T, V), y.view(B * T))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        elapsed = time.time() - start

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{LSTM_EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    print(f"\n[LSTM] Best loss: {best_loss:.4f} | Saved to saved_models/lstm.pt")

    # Load best weights and generate sample
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "lstm.pt"), map_location=DEVICE))
    seed_ids = tokenizer.encode("language models")[:5]
    generated = model.generate(seed_ids, tokenizer, num_words=15, temperature=0.8, device=str(DEVICE))
    print(f"[LSTM Sample] {generated}")

    return model


# ───────────────────────────────────────────────────────────
# STEP 4 — TRANSFORMER MODEL
# ───────────────────────────────────────────────────────────

def train_transformer(tokenizer, token_ids):
    separator("STEP 4: Training Mini Transformer (GPT-style)")

    # Build data loader
    dataloader = get_dataloader(token_ids, SEQ_LEN, BATCH_SIZE)

    # Initialize model
    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=TRANS_LAYERS,
        max_seq_len=SEQ_LEN + 1,
        dropout=0.1
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    print(f"\nTraining for {TRANS_EPOCHS} epochs on {DEVICE}...")

    best_loss = float("inf")
    for epoch in range(1, TRANS_EPOCHS + 1):
        start = time.time()
        model.train()
        total_loss = 0

        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()

            logits = model(x)  # (B, T, vocab_size)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        elapsed = time.time() - start

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "transformer.pt"))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{TRANS_EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

    print(f"\n[Transformer] Best loss: {best_loss:.4f} | Saved to saved_models/transformer.pt")

    # Load best weights and generate sample
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "transformer.pt"), map_location=DEVICE))
    seed_ids = tokenizer.encode("the transformer model")[:5]
    generated = model.generate(seed_ids, tokenizer, num_words=15, temperature=0.8, device=str(DEVICE))
    print(f"[Transformer Sample] {generated}")

    return model


# ───────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 MiniLLM — Training All Models")
    print(f"   Device: {DEVICE}")
    print(f"   Data:   {DATA_PATH}")

    t0 = time.time()

    tokenizer, token_ids, token_words = prepare_data()
    train_ngram(token_words)
    train_lstm(tokenizer, token_ids)
    train_transformer(tokenizer, token_ids)

    total_time = time.time() - t0
    separator(f"✅ ALL MODELS TRAINED in {total_time:.1f}s")
    print("  Saved to: ./saved_models/")
    print("  → ngram.pkl")
    print("  → lstm.pt")
    print("  → transformer.pt")
    print("  → vocab.json")
    print("\n  Run: streamlit run app.py")
