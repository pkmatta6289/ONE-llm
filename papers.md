# ONE-LLM Essential Reading List

To truly understand the architecture we're building (which matches the modern LLaMA/Gemma family), here are the foundational research papers you should read. They map directly to the components in our `math.md`:

### 1. The Core Foundation
* **Attention Is All You Need** (Vaswani et al., 2017)
  * **Why read it:** This is the genesis of the Transformer. You need to understand standard Multi-Head Attention, residual connections, and the overall encoder-decoder structure (even though we are only building the decoder).
  * **Link:** https://arxiv.org/abs/1706.03762

### 2. Normalization (RMSNorm)
* **Root Mean Square Layer Normalization** (Zhang and Sennrich, 2019)
  * **Why read it:** Explains why dropping the mean-centering from LayerNorm doesn't hurt performance and how it speeds up computation. 
  * **Link:** https://arxiv.org/abs/1910.07467

### 3. Positional Embeddings (RoPE)
* **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
  * **Why read it:** Introduces RoPE. The math can be heavy, but the core idea of using complex rotations to encode relative positions is brilliant and is now the industry standard (used in LLaMA, Mistral, Gemma, etc.).
  * **Link:** https://arxiv.org/abs/2104.09864

### 4. Feed-Forward Network (SwiGLU)
* **GLU Variants Improve Transformer** (Shazeer, 2020)
  * **Why read it:** A very short, practical paper by Noam Shazeer (one of the original Transformer authors) showing that replacing the standard ReLU FFN with SwiGLU (Swish-Gated Linear Unit) yields significantly better performance.
  * **Link:** https://arxiv.org/abs/2002.05202

### 5. Memory-Efficient Attention (GQA)
* **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023)
  * **Why read it:** Explains the transition from Multi-Head Attention (MHA) to Multi-Query Attention (MQA), and introduces Grouped Query Attention (GQA) as the perfect middle ground for fast inference and high accuracy.
  * **Link:** https://arxiv.org/abs/2305.13245

### 🏆 Bonus: The Blueprint
* **LLaMA: Open and Efficient Foundation Language Models** (Touvron et al., 2023)
  * **Why read it:** This paper brought all these specific pieces (RMSNorm, RoPE, SwiGLU) together into one single architecture. The model we are building is essentially a miniature LLaMA!
  * **Link:** https://arxiv.org/abs/2302.13971
