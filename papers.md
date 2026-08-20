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

---

## Important Papers to Learn in the Future

Beyond the core architecture of ONE-LLM, these are the fundamental papers across modern AI that you should study next:

### Foundational Architectures
* **Attention Is All You Need** (Transformer)
  * **Link:** https://arxiv.org/abs/1706.03762
* **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**
  * **Link:** https://arxiv.org/abs/1810.04805
* **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (Vision Transformer / ViT)
  * **Link:** https://arxiv.org/abs/2010.11929

### Fine-Tuning & Parameter Efficiency (PEFT)
* **LoRA: Low-Rank Adaptation of Large Language Models**
  * **Link:** https://arxiv.org/abs/2106.09685
* **PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods** 
  * **Link:** https://github.com/huggingface/peft (Concepts library)

### Alignment & Generation
* **Training language models to follow instructions with human feedback** (InstructGPT / RLHF)
  * **Link:** https://arxiv.org/abs/2203.02155
* **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (RAG)
  * **Link:** https://arxiv.org/abs/2005.11401

### Advanced Scaling & Routing
* **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** (MoE)
  * **Link:** https://arxiv.org/abs/1701.06538

### Generative Paradigms
* **Generative Adversarial Nets** (GAN)
  * **Link:** https://arxiv.org/abs/1406.2661
* **Denoising Diffusion Probabilistic Models** (Diffusion)
  * **Link:** https://arxiv.org/abs/2006.11239
