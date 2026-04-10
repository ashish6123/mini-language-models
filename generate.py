# generate.py
# Load trained models and generate text from a prompt.
# Usage: python generate.py --model transformer --prompt "language models" --words 30

import argparse
import os
import torch
from utils.tokenizer import Tokenizer
from models.ngram import NGramModel
from models.lstm import LSTMLanguageModel
from models.transformer import MiniTransformer

SAVE_DIR = "saved_models"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model hyperparameters (must match train.py) ──────────────
EMBED_DIM   = 64
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
D_MODEL     = 64
NUM_HEADS   = 4
TRANS_LAYERS = 2
SEQ_LEN     = 32


def load_tokenizer():
    """Load saved vocabulary."""
    vocab_path = os.path.join(SAVE_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError("Vocabulary not found. Run train.py first.")
    tokenizer = Tokenizer()
    tokenizer.load(vocab_path)
    return tokenizer


def generate_ngram(prompt, num_words, temperature):
    """Generate text using the N-gram model."""
    model_path = os.path.join(SAVE_DIR, "ngram.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("N-gram model not found. Run train.py first.")

    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(SAVE_DIR, "vocab.json"))

    ngram = NGramModel()
    ngram.load(model_path)

    seed_tokens = tokenizer.tokenize(prompt)
    output = ngram.generate(seed_tokens, num_words=num_words, temperature=temperature)
    return output


def generate_lstm(prompt, num_words, temperature):
    """Generate text using the LSTM model."""
    model_path = os.path.join(SAVE_DIR, "lstm.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError("LSTM model not found. Run train.py first.")

    tokenizer = load_tokenizer()

    model = LSTMLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    seed_ids = tokenizer.encode(prompt)
    if not seed_ids:
        seed_ids = [1]  # fallback to <UNK>

    output = model.generate(seed_ids, tokenizer, num_words=num_words,
                            temperature=temperature, device=str(DEVICE))
    return output


def generate_transformer(prompt, num_words, temperature):
    """Generate text using the Transformer model."""
    model_path = os.path.join(SAVE_DIR, "transformer.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Transformer model not found. Run train.py first.")

    tokenizer = load_tokenizer()

    model = MiniTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=TRANS_LAYERS,
        max_seq_len=SEQ_LEN + 1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    seed_ids = tokenizer.encode(prompt)
    if not seed_ids:
        seed_ids = [1]

    output = model.generate(seed_ids, tokenizer, num_words=num_words,
                            temperature=temperature, device=str(DEVICE))
    return output


def main():
    parser = argparse.ArgumentParser(description="MiniLLM Text Generator")
    parser.add_argument("--model", choices=["ngram", "lstm", "transformer"],
                        default="transformer", help="Model to use for generation")
    parser.add_argument("--prompt", type=str, default="language models",
                        help="Seed text to start generation")
    parser.add_argument("--words", type=int, default=30,
                        help="Number of words to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0.1=deterministic, 2.0=creative)")
    args = parser.parse_args()

    print(f"\n🤖 MiniLLM Generator")
    print(f"   Model:       {args.model}")
    print(f"   Prompt:      '{args.prompt}'")
    print(f"   Words:       {args.words}")
    print(f"   Temperature: {args.temperature}")
    print("-" * 50)

    if args.model == "ngram":
        output = generate_ngram(args.prompt, args.words, args.temperature)
    elif args.model == "lstm":
        output = generate_lstm(args.prompt, args.words, args.temperature)
    else:
        output = generate_transformer(args.prompt, args.words, args.temperature)

    print(f"\n📝 Generated Text:\n{output}\n")


if __name__ == "__main__":
    main()
