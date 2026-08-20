# ONE-LLM Architecture & Training Walkthrough

We successfully implemented an entire modern LLM (LLaMA architecture) from scratch in pure PyTorch, and trained it locally on your MacBook Air!

## What We Built
1. **Rotary Position Embeddings (RoPE):** Complex-space rotations to perfectly encode relative positional distance without fixed positional embeddings.
2. **Grouped Query Attention (GQA):** Memory-efficient attention mechanism that drastically reduces KV-cache memory requirements.
3. **SwiGLU FFN:** Multiplicative gating for better capacity than standard ReLU networks.
4. **RMSNorm & Pre-normalization:** Faster, more stable normalization layers.
5. **Custom PyTorch Training Loop:** Utilized the `AdamW` optimizer, cosine learning rate schedules, and Apple's MPS (Metal Performance Shaders) backend for hardware acceleration.

## Training Results
We configured a ~17.8 Million parameter "Micro" model (14.5M embeddings + 3.3M transformer parameters) and trained it on a text dataset to prove the pipeline works end-to-end.

**Key Metrics from the Run:**
- **Hardware:** Apple Silicon GPU (`mps`)
- **Training Throughput:** Peak of **~8,157 tokens per second**
- **Loss trajectory:** Dropped from `10.86` to `5.85` in just 100 steps (Perplexity: 348).
- **Optimizer:** AdamW with a peak learning rate of `6e-4`.

---

## 📄 CV / Resume Bullet Points

Because you built this yourself and have actual data to back it up, you can add this to the "Projects" section of your CV. Here are a few ways to phrase it (choose the one that best fits your resume's style):

**Option 1 (Hardware & Performance Focused):**
> * **ONE-LLM (PyTorch):** Engineered a complete ~18M parameter autoregressive Large Language Model from scratch using the LLaMA architecture (RoPE, GQA, SwiGLU, RMSNorm).
> * Implemented custom data pipelines and training loops, utilizing Apple's Metal Performance Shaders (MPS) to achieve training throughputs exceeding 8,100 tokens/second on edge hardware.

**Option 2 (Architecture Focused):**
> * **Custom LLM Implementation:** Built a from-scratch decoder-only transformer model in PyTorch without relying on high-level abstractions like HuggingFace.
> * Implemented modern architectural advancements including Grouped Query Attention (GQA) for memory efficiency, Rotary Position Embeddings (RoPE) for context modeling, and SwiGLU gating for improved feed-forward capacity.
> * Developed a custom Byte-Pair Encoding (BPE) tokenizer and end-to-end training pipeline utilizing AdamW and cosine learning rate schedules.

**Option 3 (Short & Punchy):**
> * **ONE-LLM:** Developed an 18-million-parameter Large Language Model from scratch in pure PyTorch based on the LLaMA architecture. Implemented RoPE, GQA, and SwiGLU, and engineered an MPS-accelerated training pipeline achieving >8k tokens/sec on local hardware.

> [!TIP]
> If an interviewer asks you about this project, be prepared to explain *why* RMSNorm is faster than LayerNorm (no mean centering), or *why* GQA is used instead of standard Multi-Head Attention (saves memory during inference by sharing Key/Value heads across multiple Query heads). You have the `math.md` file to review these concepts!
