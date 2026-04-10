# app.py
# Streamlit web UI for MiniLLM text generation.
# Run with: streamlit run app.py

import os
import sys
import torch
import streamlit as st

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="MiniLLM — Unsupervised Language Models",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    .output-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #e2e8f0;
        margin-top: 1rem;
    }

    .info-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border: 1px solid #334155;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Lazy model loaders (cached) ────────────────────────────

SAVE_DIR = "saved_models"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM    = 64
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
D_MODEL      = 64
NUM_HEADS    = 4
TRANS_LAYERS = 2
SEQ_LEN      = 32


@st.cache_resource(show_spinner=False)
def load_tokenizer():
    from utils.tokenizer import Tokenizer
    tok = Tokenizer()
    tok.load(os.path.join(SAVE_DIR, "vocab.json"))
    return tok


@st.cache_resource(show_spinner=False)
def load_ngram():
    from models.ngram import NGramModel
    m = NGramModel()
    m.load(os.path.join(SAVE_DIR, "ngram.pkl"))
    return m


@st.cache_resource(show_spinner=False)
def load_lstm():
    from models.lstm import LSTMLanguageModel
    tok = load_tokenizer()
    m = LSTMLanguageModel(tok.vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(SAVE_DIR, "lstm.pt"), map_location=DEVICE))
    m.eval()
    return m


@st.cache_resource(show_spinner=False)
def load_transformer():
    from models.transformer import MiniTransformer
    tok = load_tokenizer()
    m = MiniTransformer(tok.vocab_size, D_MODEL, NUM_HEADS, TRANS_LAYERS, SEQ_LEN + 1).to(DEVICE)
    m.load_state_dict(torch.load(os.path.join(SAVE_DIR, "transformer.pt"), map_location=DEVICE))
    m.eval()
    return m


def check_models_trained():
    required = ["vocab.json", "ngram.pkl", "lstm.pt", "transformer.pt"]
    return all(os.path.exists(os.path.join(SAVE_DIR, f)) for f in required)


def generate_text(model_choice, prompt, num_words, temperature):
    """Route generation to the selected model."""
    tokenizer = load_tokenizer()

    if model_choice == "N-Gram (Trigram)":
        ngram = load_ngram()
        seed_tokens = tokenizer.tokenize(prompt) or ["the"]
        return ngram.generate(seed_tokens, num_words=num_words, temperature=temperature)

    elif model_choice == "LSTM":
        model = load_lstm()
        seed_ids = tokenizer.encode(prompt) or [1]
        return model.generate(seed_ids, tokenizer, num_words=num_words,
                               temperature=temperature, device=str(DEVICE))

    else:  # Transformer
        model = load_transformer()
        seed_ids = tokenizer.encode(prompt) or [1]
        return model.generate(seed_ids, tokenizer, num_words=num_words,
                               temperature=temperature, device=str(DEVICE))


# ── SIDEBAR ────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        "🤖 Select Model",
        ["N-Gram (Trigram)", "LSTM", "Transformer (GPT-style)"],
        help="Choose which language model to use for generation."
    )

    num_words = st.slider(
        "📝 Words to Generate", min_value=10, max_value=80, value=30, step=5,
        help="How many new words to generate."
    )

    temperature = st.slider(
        "🌡️ Temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.1,
        help="Lower = more focused. Higher = more creative/random."
    )

    st.markdown("---")
    st.markdown("### 📊 Architecture Info")

    if model_choice == "N-Gram (Trigram)":
        st.markdown("""<div class='info-card'>
        <b>Trigram Model</b><br>
        Uses previous 2 words to predict the next.<br>
        Fast, interpretable, no neural network.
        </div>""", unsafe_allow_html=True)
    elif model_choice == "LSTM":
        st.markdown(f"""<div class='info-card'>
        <b>LSTM Network</b><br>
        Embed: {EMBED_DIM}d → LSTM: {HIDDEN_DIM}h × {NUM_LAYERS}L<br>
        Captures sequential dependencies.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='info-card'>
        <b>Mini Transformer (GPT)</b><br>
        d_model={D_MODEL} | heads={NUM_HEADS} | layers={TRANS_LAYERS}<br>
        Self-attention over full context.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"🖥️ Device: `{DEVICE}`")
    st.caption("Built with PyTorch + Streamlit")


# ── MAIN CONTENT ───────────────────────────────────────────

st.markdown('<div class="main-title">🧠 MiniLLM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Unsupervised Learning of Language Models — built from scratch</div>',
            unsafe_allow_html=True)

# Check if models are trained
if not check_models_trained():
    st.error("⚠️ Models not found! Please train first:")
    st.code("python train.py", language="bash")
    st.stop()

# ── Input area ────────────────────────────────────────────
st.markdown("#### 💬 Enter a Prompt")
prompt = st.text_input(
    label="prompt_input",
    value="language models learn from",
    placeholder="Type your seed text here...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 1])
with col1:
    generate_btn = st.button("✨ Generate Text", use_container_width=True)
with col2:
    st.markdown(f"<br>", unsafe_allow_html=True)

# ── Generation ────────────────────────────────────────────
if generate_btn:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner(f"Generating with {model_choice}..."):
            try:
                output = generate_text(model_choice, prompt.strip(), num_words, temperature)

                st.markdown("#### 📝 Generated Output")
                st.markdown(f'<div class="output-box">{output}</div>', unsafe_allow_html=True)

                # Stats
                word_count = len(output.split())
                char_count = len(output)
                st.markdown(f"""
                <div style="display:flex; gap:2rem; margin-top:0.8rem; color:#64748b; font-size:0.85rem;">
                    <span>📊 <b>{word_count}</b> words</span>
                    <span>🔤 <b>{char_count}</b> chars</span>
                    <span>🌡️ temp=<b>{temperature}</b></span>
                    <span>🤖 <b>{model_choice}</b></span>
                </div>
                """, unsafe_allow_html=True)

                # Copy button
                st.text_area("Copy output:", output, height=100, label_visibility="collapsed")

            except Exception as e:
                st.error(f"Generation error: {e}")
                st.info("Make sure you ran `python train.py` first.")

# ── Model comparison section ──────────────────────────────
st.markdown("---")
with st.expander("📚 How Each Model Works"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**N-Gram**")
        st.markdown("""
        - Counts word sequences in training data
        - P(next | prev N-1 words)
        - Fast, no training loop
        - Limited long-range context
        """)

    with col2:
        st.markdown("**LSTM**")
        st.markdown("""
        - Embedding → LSTM → Linear
        - Hidden state carries memory
        - Learns sequential patterns
        - Trained via backprop
        """)

    with col3:
        st.markdown("**Transformer**")
        st.markdown("""
        - Self-attention over all tokens
        - Positional encoding
        - Multi-head attention
        - State-of-the-art architecture
        """)

with st.expander("🎓 About This Project"):
    st.markdown("""
    **MiniLLM** demonstrates **unsupervised (self-supervised) learning** for language modeling.

    All three models are trained on raw text with **no labels** — the training signal comes from
    predicting the next word given previous words. This is the same principle behind GPT, LLaMA, and other LLMs.

    **Training objective:** Given tokens `[w₁, w₂, ..., wₙ]`, predict `wₙ₊₁`.
    Loss = Cross-Entropy between predicted distribution and true next token.

    Built entirely with **PyTorch** — no HuggingFace, no pretrained weights.
    """)
