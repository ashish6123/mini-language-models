# 🧠 MiniLLM: Unsupervised Learning of Language Models from Scratch

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<p align="center">
  <b>Three language models — N-gram, LSTM, Transformer — trained from scratch using unsupervised learning.</b><br>
  One command to train. One command to launch the UI.
</p>

---

## 📌 Project Overview

**MiniLLM** is an educational, end-to-end implementation of three language modeling approaches, all trained using **self-supervised (unsupervised) learning** on raw text — no labels required.

| Model | Architecture | Context | Speed |
|-------|-------------|---------|-------|
| **N-Gram** | Probability tables | Fixed (n-1 words) | Instant |
| **LSTM** | Embedding → LSTM → Linear | Sequential memory | Fast |
| **Transformer** | Multi-head self-attention | Full window | Best quality |

The core training objective for all neural models:

> Given a sequence of words `[w₁, w₂, ..., wₙ]`, predict `wₙ₊₁` using **cross-entropy loss**.

This is the same principle that powers GPT-4, LLaMA, Claude, and all modern LLMs.

---

## 🗂️ Project Structure

```
mini-llm/
│
├── data/
│   └── sample.txt              ← Training corpus
│
├── models/
│   ├── ngram.py                ← Trigram model with Laplace smoothing
│   ├── lstm.py                 ← LSTM language model
│   └── transformer.py          ← GPT-style transformer (from scratch)
│
├── utils/
│   ├── tokenizer.py            ← Word-level tokenizer with vocab builder
│   └── dataset.py              ← PyTorch dataset for next-token prediction
│
├── saved_models/               ← Auto-created after training
│   ├── vocab.json
│   ├── ngram.pkl
│   ├── lstm.pt
│   └── transformer.pt
│
├── train.py                    ← Train all 3 models
├── generate.py                 ← CLI text generation
├── app.py                      ← Streamlit web UI
├── requirements.txt
├── run.sh                      ← One-command setup + train + launch
└── README.md
```

---

## 🚀 Quick Start

### Option 1: One Command (Recommended)

```bash
bash run.sh
```

This will:
1. Create a Python virtual environment
2. Install all dependencies
3. Train all 3 models (~1–3 min on CPU)
4. Launch the Streamlit app at `http://localhost:8501`

### Option 2: Manual Step-by-Step

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train all models
python train.py

# 4. Launch the web UI
streamlit run app.py

# Optional: CLI generation
python generate.py --model transformer --prompt "neural networks" --words 30
```

---

## ⚙️ How It Works

### 1. Tokenization
Raw text is split into word tokens. A vocabulary is built mapping each unique word to an integer ID. The tokenizer supports `encode()` (text → IDs) and `decode()` (IDs → text).

### 2. Dataset Pipeline
The encoded token sequence is sliced into fixed-length windows:
- **Input**: `[w₁, w₂, ..., w₃₂]`
- **Target**: `[w₂, w₃, ..., w₃₃]` (shifted by 1)

This teaches the model: *given these tokens, predict the next one at every position.*

### 3. N-Gram Model
- Counts how often each trigram `(w₁, w₂, w₃)` appears in training text
- At generation time: given the last 2 words, sample the next word by probability
- Uses Laplace (add-1) smoothing for unseen contexts

### 4. LSTM Model
```
Input IDs → Embedding (64d) → LSTM (128h × 2L) → Dropout → Linear → Logits
```
- The LSTM's hidden state acts as a compressed memory of the sequence
- Trained with cross-entropy loss + Adam optimizer + gradient clipping

### 5. Transformer Model (GPT-style)
```
Input IDs → Token Embedding + Positional Encoding
         → [Multi-Head Self-Attention → Add&Norm → FFN → Add&Norm] × 2
         → LayerNorm → Linear → Logits
```
- **Causal masking**: position `i` can only attend to positions `0..i`
- **Multi-head attention**: 4 heads, each learning different relationships
- Fully implemented from scratch — no HuggingFace

---

## 💻 Streamlit UI

The web app at `http://localhost:8501` provides:

- **Prompt input**: Enter any seed text
- **Model selector**: Switch between N-Gram, LSTM, Transformer
- **Temperature slider**: Control creativity (0.1 = focused, 2.0 = random)
- **Word count slider**: Choose how many words to generate
- **Generated output**: Displayed with word/character stats

---

## 📊 Sample Outputs

**Prompt**: `"language models learn"`

| Model | Output |
|-------|--------|
| N-Gram | `language models learn to predict the next word in a sequence by training on large amounts` |
| LSTM | `language models learn linguistic patterns and semantic representations from raw text data` |
| Transformer | `language models learn to represent the structure of natural language through self-supervised objectives` |

---

## 🎛️ CLI Usage

```bash
# Generate with transformer
python generate.py --model transformer --prompt "deep learning" --words 25 --temperature 0.8

# Generate with LSTM
python generate.py --model lstm --prompt "neural networks" --words 20 --temperature 1.0

# Generate with N-gram
python generate.py --model ngram --prompt "the model" --words 20 --temperature 0.9
```

---

## 🧪 Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning Rate | 3e-3 |
| Scheduler | StepLR (γ=0.8, step=5) |
| Epochs | 20 |
| Batch Size | 16 |
| Sequence Length | 32 |
| Gradient Clipping | 1.0 |

---

## 📸 Screenshots

> *[Screenshot: Streamlit UI with transformer output]*
> *[Screenshot: Training logs in terminal]*
> *[Screenshot: Model architecture sidebar]*

---

## 🧠 Key Concepts Demonstrated

- **Unsupervised / Self-Supervised Learning**: No human-labeled data needed
- **Next-Token Prediction**: The universal language modeling objective
- **Causal Self-Attention**: Core mechanism behind GPT-family models
- **Positional Encoding**: Injecting order into attention-based models
- **Temperature Sampling**: Controlling randomness in generation
- **Gradient Clipping**: Preventing exploding gradients in deep networks

---

## 📚 References

- Vaswani et al. (2017) — *Attention Is All You Need*
- Radford et al. (2018) — *Improving Language Understanding by Generative Pre-Training*
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*

---

## 👤 Author

Built as a research showcase project for **Unsupervised Learning of Large Language Models** (IIT Jammu RiSE UP 2026).

---

## 📄 License

MIT License — free to use, modify, and distribute.
