# ONE-LLM — 7-Day Milestone Plan

**Start:** August 17 (Sunday)  
**Target:** August 23 (Saturday)  
**Daily commitment:** ~4-6 hours focused work

---

## Day 1 (Sun Aug 17) — BPE Tokenizer 🔤
**Goal:** Build a working byte-level BPE tokenizer from scratch.

### Deliverables
| File | What to Build |
|------|--------------|
| `tokenizer/bpe.py` | BPE training: count pairs, merge most frequent, repeat |
| `tokenizer/tokenizer.py` | Encode text → token IDs, decode IDs → text, special tokens |
| `tokenizer/train_tokenizer.py` | CLI script to train on a text file |

### Done When ✅
- [ ] You can train BPE on a small text file (e.g., a book from Project Gutenberg)
- [ ] `encode("hello world")` → list of ints → `decode()` back to `"hello world"` perfectly
- [ ] Vocab size is configurable (e.g., 512, 4096, 32000)
- [ ] Special tokens (`<|bos|>`, `<|eos|>`, `<|pad|>`) work
- [ ] You can save/load a trained tokenizer

### Key Concepts to Understand
1. **Why bytes?** Start with 256 base tokens (one per byte). This means ANY text can be tokenized — no `<unk>` ever
2. **The merge loop:** Count all adjacent pairs → merge the most frequent → repeat N times = N merges = 256 + N vocab size
3. **Regex pre-tokenization:** GPT-2 splits text with a regex BEFORE BPE to prevent merges across word boundaries. Pattern: `r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""`
4. **Encoding vs Training:** Training finds the merges. Encoding applies them greedily in priority order

---

## Day 2 (Mon Aug 18) — Model Foundations 🧱
**Goal:** Build the three core primitives: config, RMSNorm, RoPE.

### Deliverables
| File | What to Build |
|------|--------------|
| `model/config.py` | `ModelConfig` dataclass with all hyperparameters |
| `model/rmsnorm.py` | RMSNorm layer |
| `model/rope.py` | `precompute_freqs_cis()` + `apply_rotary_emb()` |

### Done When ✅
- [ ] `RMSNorm` produces correct shapes, has learnable `γ` (gain) parameter
- [ ] You can explain why RMSNorm drops mean-centering and why that's fine
- [ ] `precompute_freqs_cis(dim=64, seq_len=128)` returns complex-valued rotation frequencies
- [ ] `apply_rotary_emb(q, k, freqs_cis)` rotates Q and K correctly
- [ ] You understand: `q·k` after rotation depends on *relative* position, not absolute

### Key Concepts to Understand
1. **RMSNorm math:** `RMS(x) = sqrt(mean(x²) + ε)`, then `x_norm = x / RMS(x) * γ`. No mean subtraction.
2. **RoPE intuition:** Pair up dimensions (d₀,d₁), (d₂,d₃), ... and rotate each pair by `θ_i * position`. Different dimensions rotate at different frequencies (like a clock with different-speed hands).
3. **Why complex numbers?** Rotation in 2D = multiply by `e^{iθ}`. PyTorch's `torch.polar` or `view_as_complex` makes this clean.

---

## Day 3 (Tue Aug 19) — Attention & FFN ⚡
**Goal:** Build GQA (the hardest single component) and SwiGLU FFN.

### Deliverables
| File | What to Build |
|------|--------------|
| `model/attention.py` | Grouped Query Attention with causal mask + KV-cache support |
| `model/feedforward.py` | SwiGLU feed-forward network |

### Done When ✅
- [ ] GQA with `n_heads=12, n_kv_heads=4` produces correct output shapes
- [ ] Causal mask works — future tokens have zero attention weight
- [ ] When `n_kv_heads == n_heads`, output matches standard MHA exactly
- [ ] KV-cache: passing `start_pos > 0` uses cached K,V and only processes new tokens
- [ ] SwiGLU has 3 linear layers (gate, up, down), uses `F.silu` for the gate

### Key Concepts to Understand
1. **GQA:** 12 Q heads but only 4 KV heads. Each KV head is shared by 3 Q heads. The key operation is `repeat_kv(k, n_rep=3)` to expand K,V before the attention dot product.
2. **Causal masking:** `mask = torch.triu(torch.full((seq, seq), -inf), diagonal=1)`. Added to attention scores before softmax.
3. **SwiGLU:** `output = (silu(x @ W_gate) * (x @ W_up)) @ W_down`. The `silu(x) = x * sigmoid(x)` gating controls information flow.

---

## Day 4 (Wed Aug 20) — Full Model + Data Pipeline 🏗️
**Goal:** Assemble the complete model. Build the data pipeline.

### Deliverables
| File | What to Build |
|------|--------------|
| `model/transformer_block.py` | Single block: norm → attn → residual → norm → ffn → residual |
| `model/model.py` | Full model: embedding → N blocks → norm → output head |
| `data/dataset.py` | Memory-mapped dataset from tokenized binary files |
| `data/dataloader.py` | Batched data loading with sequence packing |
| `scripts/download_data.py` | Download + tokenize a small dataset |

### Done When ✅
- [ ] `model = OneLLM(config)` — you can print param count (~124M for small config)
- [ ] Forward pass: `logits = model(token_ids)` returns `(batch, seq_len, vocab_size)`
- [ ] Weight tying: embedding weights == output projection weights
- [ ] Dataset loads from a `.bin` file of pre-tokenized uint16 token IDs
- [ ] DataLoader yields `(batch, seq_len)` chunks correctly

### Key Concepts to Understand
1. **Pre-norm:** Norm BEFORE attention/FFN (not after). This is more stable for training deep networks.
2. **Weight tying:** The embedding matrix `(vocab_size, d_model)` is reused as the output projection. Saves params, acts as a regularizer.
3. **Memory mapping:** `np.memmap` lets you access a huge file as if it's in RAM, but only pages in what you read. Essential for large datasets.

---

## Day 5 (Thu Aug 21) — Pretraining 🔥
**Goal:** Train the model. See the loss go down.

### Deliverables
| File | What to Build |
|------|--------------|
| `training/optimizer.py` | AdamW with weight decay filtering |
| `training/lr_schedule.py` | Linear warmup + cosine decay |
| `training/pretrain.py` | Full training loop with mixed precision + grad accumulation |
| `training/utils.py` | Checkpointing, logging |

### Done When ✅
- [ ] Training loop runs without crashing on a small dataset
- [ ] Loss decreases steadily (overfit a tiny dataset to near-zero loss first!)
- [ ] Mixed precision (`bfloat16` / `float16`) works and speeds up training
- [ ] Gradient accumulation simulates larger batch sizes
- [ ] Checkpoints save/load correctly (model + optimizer + step count)
- [ ] You see: learning rate warmup → peak → cosine decay in your logs

### Key Concepts to Understand
1. **AdamW vs Adam:** AdamW decouples weight decay from gradient updates. Apply decay only to 2D weight tensors, NOT to biases, norms, or embeddings.
2. **Cosine schedule:** `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * step / total_steps))`
3. **Gradient accumulation:** Sum gradients over N mini-batches before calling `optimizer.step()`. Effective batch = N × mini-batch.
4. **Mixed precision:** `torch.autocast` computes in float16/bfloat16 for speed, but keeps master weights in float32 for stability.

---

## Day 6 (Fri Aug 22) — Inference + SFT 🗣️
**Goal:** Generate text. Fine-tune on instructions.

### Deliverables
| File | What to Build |
|------|--------------|
| `inference/generate.py` | Autoregressive generation with KV-cache |
| `inference/sampling.py` | Temperature, top-k, top-p, repetition penalty |
| `inference/chat.py` | Interactive terminal chat |
| `training/sft.py` | Supervised fine-tuning on instruction data |

### Done When ✅
- [ ] Model generates coherent (or at least grammatical) text
- [ ] KV-cache makes generation fast (only 1 token processed per step, not the whole sequence)
- [ ] Temperature=0 gives deterministic output, temperature>1 gives diverse output
- [ ] Top-p=0.9 filters to the top 90% probability mass
- [ ] SFT trains on (instruction, response) pairs with loss only on the response

### Key Concepts to Understand
1. **KV-cache:** During generation, cache K,V at each layer. For token N+1, only compute Q for the new token and attend to all cached K,V.
2. **Top-p (nucleus) sampling:** Sort logits descending, compute cumulative probability, mask everything below the p threshold.
3. **SFT loss masking:** Given `[BOS] instruction [SEP] response [EOS]`, only compute cross-entropy loss on the `response [EOS]` tokens.

---

## Day 7 (Sat Aug 23) — DPO + Exploration 🚀
**Goal:** Align the model with preferences. Explore MoE/Mamba.

### Deliverables
| File | What to Build |
|------|--------------|
| `training/dpo.py` | Direct Preference Optimization |
| `model/moe.py` | Mixture of Experts layer (stretch goal) |
| `exploration/mamba.py` | Simplified Mamba block (stretch goal) |

### Done When ✅
- [ ] DPO training loop runs on (chosen, rejected) pairs
- [ ] You understand the DPO loss: why it doesn't need a reward model
- [ ] (Stretch) MoE replaces FFN with routed experts + load-balancing loss
- [ ] (Stretch) Mamba block processes a sequence in O(n) instead of O(n²)

---

## Daily Rhythm

```
┌─────────────────────────────────────────────────┐
│  1. Read the day's concepts (30 min)            │
│  2. Write the code yourself (2-3 hours)         │
│  3. Test & debug (1 hour)                       │
│  4. Ask me when stuck — I'll guide, not solve   │
│  5. Commit to git at end of day                 │
└─────────────────────────────────────────────────┘
```

> [!TIP]
> **The #1 trap:** Don't copy-paste. Type every line. When something is confusing, stop and ask me to explain the *math* — that's where the real learning happens.

> [!IMPORTANT]
> **Overfit first, scale second.** On Day 5, start by training on a TINY dataset (a few KB of text). The model should memorize it perfectly (loss → ~0). Only then move to real data. If you can't overfit a tiny dataset, something is broken.
