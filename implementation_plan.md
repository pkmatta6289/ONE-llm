# ONE-LLM: Build a Modern LLM from Scratch — First Principles

You've built GPT-2 from tutorials. Now we build something **better** — a modern decoder-only LLM incorporating every major architectural advance since GPT-2 (2019), implemented from first principles in pure PyTorch, trainable on a single GPU.

## What Makes This Different from Your GPT-2

| GPT-2 (What You Built) | ONE-LLM (What We'll Build) |
|---|---|
| Learned absolute positional embeddings | **RoPE** (Rotary Position Embeddings) — extrapolates to longer sequences |
| LayerNorm (post-norm) | **RMSNorm** (pre-norm) — simpler, faster, more stable |
| Standard Multi-Head Attention | **Grouped Query Attention (GQA)** — drastically reduces KV-cache memory |
| GELU FFN | **SwiGLU** FFN — better gradient flow, used in LLaMA/Gemma |
| No KV-Cache | **KV-Cache** for fast autoregressive inference |
| Basic Adam | **AdamW** with cosine annealing, warmup, gradient clipping, weight decay |
| Simple data loading | **Custom BPE tokenizer** trained from scratch |
| Just pretraining | **Pretraining → SFT → DPO** full pipeline |
| Single architecture | Optional **Mixture of Experts (MoE)** and **Mamba/SSM** exploration |

## Architecture: "ONE-LLM" (~124M → 300M params)

The model follows the LLaMA / Gemma family design, which represents the modern consensus:

```
Input Token IDs
      │
  Embedding (no position embedding — RoPE is applied in attention)
      │
  ┌───────────────────────────────────────────────┐
  │              Transformer Block (×N)            │
  │                                                │
  │   RMSNorm → GQA Self-Attention (with RoPE)     │
  │       │ + residual                             │
  │   RMSNorm → SwiGLU FFN                         │
  │       │ + residual                             │
  └───────────────────────────────────────────────┘
      │
  RMSNorm (final)
      │
  Linear (to vocab) — weight-tied with Embedding
      │
  Logits
```

### Model Configurations

| Config | Layers | d_model | Heads | KV Heads | FFN dim | Params |
|--------|--------|---------|-------|----------|---------|--------|
| **Small** | 12 | 768 | 12 | 4 (GQA) | 2048 | ~124M |
| **Medium** | 24 | 1024 | 16 | 4 (GQA) | 2816 | ~300M |

---

## Project Structure

```
ONE-llm/
├── README.md
├── requirements.txt
│
├── tokenizer/                  # Phase 1: BPE Tokenizer
│   ├── __init__.py
│   ├── bpe.py                  # BPE training algorithm from scratch
│   ├── tokenizer.py            # Encode/decode, special tokens, vocab
│   └── train_tokenizer.py      # Script to train on data
│
├── model/                      # Phase 2: Model Architecture
│   ├── __init__.py
│   ├── config.py               # Model configuration dataclass
│   ├── embedding.py            # Token embedding (no positional)
│   ├── rope.py                 # Rotary Position Embeddings
│   ├── rmsnorm.py              # RMSNorm
│   ├── attention.py            # Grouped Query Attention + KV-Cache
│   ├── feedforward.py          # SwiGLU FFN
│   ├── transformer_block.py    # Single transformer block
│   ├── model.py                # Full ONE-LLM model
│   └── moe.py                  # (Optional) Mixture of Experts layer
│
├── data/                       # Phase 3: Data Pipeline
│   ├── __init__.py
│   ├── dataset.py              # Memory-mapped dataset, chunking
│   └── dataloader.py           # Custom dataloader with packing
│
├── training/                   # Phase 4: Training
│   ├── __init__.py
│   ├── pretrain.py             # Pretraining loop
│   ├── sft.py                  # Supervised fine-tuning
│   ├── dpo.py                  # Direct Preference Optimization
│   ├── optimizer.py            # AdamW with cosine schedule, warmup
│   ├── lr_schedule.py          # Learning rate schedulers
│   └── utils.py                # Checkpointing, logging, metrics
│
├── inference/                  # Phase 5: Inference
│   ├── __init__.py
│   ├── generate.py             # Text generation with KV-cache
│   ├── sampling.py             # Top-k, Top-p, temperature, repetition penalty
│   └── chat.py                 # Interactive chat interface
│
├── exploration/                # Phase 6: Beyond Transformers
│   ├── mamba.py                # Mamba / S4 / SSM exploration
│   └── README.md               # Notes on alternative architectures
│
├── scripts/
│   ├── download_data.py        # Download and prepare training data
│   ├── benchmark.py            # Benchmarking and evaluation
│   └── visualize.py            # Attention visualization, loss curves
│
├── configs/
│   ├── small.yaml              # 124M config
│   └── medium.yaml             # 300M config
│
├── tests/
│   ├── test_tokenizer.py
│   ├── test_rope.py
│   ├── test_attention.py
│   ├── test_model.py
│   └── test_generation.py
│
└── notebooks/                  # (Optional) Jupyter exploration
    └── understanding_rope.ipynb
```

---

## Implementation Phases

### Phase 1: BPE Tokenizer from Scratch
**Why this matters:** Most tutorials use `tiktoken` or HuggingFace tokenizers. Building BPE yourself teaches you how vocabulary is created, why subword tokenization works, and what "tokens" really are.

**What we'll implement:**
1. **Byte-level BPE** — Start with individual bytes (256 base tokens), iteratively merge the most frequent adjacent pairs
2. **Training loop** — Count pair frequencies, merge top pair, update corpus, repeat for N merges
3. **Encoding** — Given text, apply learned merges in priority order to produce token IDs
4. **Decoding** — Reverse: token IDs → bytes → text (handling UTF-8)
5. **Special tokens** — `<|bos|>`, `<|eos|>`, `<|pad|>`, `<|unk|>`
6. **Regex pre-tokenization** — Split on whitespace/punctuation boundaries before BPE (like GPT-2's pattern)
7. **Vocab serialization** — Save/load trained tokenizer

**Key insight vs GPT-2:** GPT-2's tokenizer uses a specific regex pattern to pre-split text. We'll implement this and understand *why* it matters (prevents cross-word merges).

---

### Phase 2: Model Architecture (The Core)

Each component is a separate file with clear mathematical derivations in comments.

#### 2a. RMSNorm (`rmsnorm.py`)
```
RMSNorm(x) = x / RMS(x) * γ
where RMS(x) = sqrt(mean(x²) + ε)
```
- **Why not LayerNorm?** RMSNorm drops the mean-centering step. Empirically just as good, ~10-15% faster, and simpler.

#### 2b. Rotary Position Embeddings (`rope.py`)
- Encode position information by rotating query/key vectors in 2D subspaces
- `q' = R(θ_pos) @ q`, `k' = R(θ_pos) @ k` where R is a rotation matrix
- The dot product `q' · k'` depends on **relative** position — no learned position embeddings needed
- Implements the `precompute_freqs_cis` + `apply_rotary_emb` pattern from LLaMA
- **Key advantage over GPT-2:** Extrapolates to longer sequences than training, encodes relative position naturally

#### 2c. Grouped Query Attention (`attention.py`)
- Standard MHA: `n_heads` Q, K, V heads each
- **GQA:** `n_heads` Q heads, but only `n_kv_heads` K and V heads (shared across Q head groups)
- When `n_kv_heads = 1` → Multi-Query Attention (MQA)
- When `n_kv_heads = n_heads` → standard MHA
- **Why?** Reduces KV-cache memory by `n_heads/n_kv_heads` × with minimal quality loss
- Implements causal masking, KV-cache for inference

#### 2d. SwiGLU FFN (`feedforward.py`)
```
SwiGLU(x) = (Swish(xW₁) ⊙ xW₃) W₂
where Swish(x) = x · σ(x)
```
- **Why not GELU?** SwiGLU has a gating mechanism (the ⊙ elementwise multiply with a parallel projection) that improves gradient flow. Used in LLaMA, Gemma, PaLM.
- Note: 3 weight matrices instead of 2, so FFN hidden dim is typically `(2/3) × 4d` to keep param count similar.

#### 2e. Transformer Block (`transformer_block.py`)
- Pre-norm architecture (norm before attention/FFN, not after)
- Residual connections
- No bias in linear layers (modern practice)

#### 2f. Full Model (`model.py`)
- Token embedding (weight-tied with output projection)
- Stack of N transformer blocks
- Final RMSNorm
- Forward pass returns logits

---

### Phase 3: Data Pipeline

**Training data:** We'll use a mix of:
- **FineWeb-Edu** (small subset, ~1-5GB) — high-quality English web text
- Downloaded and tokenized into memory-mapped binary files for efficient loading

**What we'll implement:**
1. **Download & preprocess** — Download data, clean, tokenize with our BPE tokenizer
2. **Memory-mapped dataset** — `np.memmap` for efficient access without loading entire dataset into RAM
3. **Sequence packing** — Pack multiple documents into single sequences (separated by `<|eos|>`) to avoid wasting padding tokens
4. **Custom DataLoader** — Handles batching, shuffling across the memmap

---

### Phase 4: Training Pipeline

#### 4a. Pretraining (`pretrain.py`)
- **Optimizer:** AdamW (β₁=0.9, β₂=0.95, ε=1e-8)
- **LR Schedule:** Linear warmup (2000 steps) → Cosine decay to 10% of peak LR
- **Gradient clipping:** Max norm = 1.0
- **Weight decay:** 0.1 (applied to 2D weight tensors only, not biases/norms)
- **Batch size:** Gradient accumulation to simulate large effective batch sizes
- **Mixed precision:** `torch.autocast` with `bfloat16` for speed
- **Logging:** Loss curves, gradient norms, learning rate, tokens/sec
- **Checkpointing:** Save model + optimizer + scheduler state periodically

#### 4b. Supervised Fine-Tuning (`sft.py`)
- Fine-tune the pretrained model on instruction-following data
- Format: `<|bos|> [instruction] \n [response] <|eos|>`
- Only compute loss on the response tokens (mask instruction tokens)
- Lower learning rate, fewer epochs

#### 4c. DPO — Direct Preference Optimization (`dpo.py`)
- **Why DPO over RLHF?** DPO is simpler (no separate reward model, no PPO), and achieves comparable results
- Takes pairs of (chosen, rejected) responses
- Loss: `L_DPO = -log σ(β (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))`
- Requires a frozen reference model (copy of the SFT model)

---

### Phase 5: Inference & Generation

1. **KV-Cache:** Cache K, V tensors from previous positions to avoid recomputation
2. **Sampling strategies:**
   - Temperature scaling
   - Top-k sampling
   - Top-p (nucleus) sampling
   - Repetition penalty
3. **Interactive chat** — Terminal-based chat interface

---

### Phase 6: Beyond Transformers (Exploration)

#### Mamba / State Space Models
- Implement a simplified Mamba block
- Understand selective state spaces — how they achieve O(n) instead of O(n²)
- Compare with transformer on a small benchmark

#### Mixture of Experts (MoE)
- Replace some FFN layers with MoE layers
- Router network selects top-k experts per token
- Load balancing auxiliary loss
- Understand why MoE gives more params with less compute

---

## Implementation Order & Timeline

| Order | Phase | What | Key Learning |
|-------|-------|------|-------------|
| 1 | **Tokenizer** | BPE from scratch | How text becomes numbers |
| 2 | **RMSNorm** | Normalization | Why pre-norm > post-norm |
| 3 | **RoPE** | Position encoding | Rotation matrices, complex numbers, relative position |
| 4 | **GQA** | Attention mechanism | KV-cache, memory efficiency, causal masking |
| 5 | **SwiGLU** | Feed-forward | Gating mechanisms, activation functions |
| 6 | **Full Model** | Assembly | Weight tying, initialization, param counting |
| 7 | **Data Pipeline** | Dataset + DataLoader | Memory mapping, sequence packing, efficient I/O |
| 8 | **Pretraining** | Training loop | AdamW, cosine schedule, mixed precision, grad accumulation |
| 9 | **Inference** | Generation | KV-cache, sampling strategies |
| 10 | **SFT** | Fine-tuning | Instruction formatting, selective loss masking |
| 11 | **DPO** | Alignment | Preference learning, reference model |
| 12 | **MoE** | Exploration | Sparse computation, routing |
| 13 | **Mamba** | Exploration | State space models, linear attention alternatives |

---

## Verification Plan

### Automated Tests
Each component gets unit tests:
```bash
python -m pytest tests/ -v
```
- **Tokenizer:** Encode/decode roundtrip, special tokens, known merges
- **RoPE:** Verify rotation properties, compare with reference implementation
- **Attention:** Compare GQA output with standard MHA (when n_kv_heads = n_heads)
- **Model:** Forward pass shape checks, gradient flow, parameter count
- **Generation:** Verify KV-cache produces identical output to naive generation

### Manual Verification
- Train on a tiny dataset first (~1MB of text) — model should overfit perfectly
- Monitor loss curves — should see smooth descent with cosine schedule
- Generate text samples at checkpoints — quality should improve visibly
- Compare perplexity with published GPT-2 small numbers on standard benchmarks

---

## Open Questions

> [!IMPORTANT]
> **Training Data:** What kind of text do you want to train on? Options:
> - **FineWeb-Edu** (general English, high quality) — recommended for pretraining
> - **A specific domain** (code, science, stories) — we can curate something
> - **Your own data** — if you have a corpus in mind

> [!IMPORTANT]
> **Compute:** What GPU do you have access to? This determines:
> - Model size (124M vs 300M)
> - Batch size and gradient accumulation steps
> - Whether we use `bfloat16` or `float16`
> - How much data we can realistically train on

> [!NOTE]
> **Depth vs Breadth:** You selected all options. I recommend we go deep on Phases 1-9 first (tokenizer → pretraining → inference), making sure every line of code is understood. Then move to SFT/DPO (Phase 10-11), and finally MoE/Mamba (Phase 12-13) as exploration. Sound right?

> [!NOTE]
> **Mathematical Derivations:** Should I include detailed mathematical derivations as comments in the code (e.g., the full derivation of why RoPE encodes relative position), or keep code clean and put derivations in separate markdown documents?
