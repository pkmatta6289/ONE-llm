# ONE-LLM Code References

If you want to study clean, minimalistic code to see how experts build these models in PyTorch, here are the absolute best references.

### 1. `llama2.c` (by Andrej Karpathy)
This is exactly the architecture we are building! Karpathy wrote a single `model.py` file that implements the LLaMA architecture (RMSNorm, RoPE, SwiGLU, GQA) in pure PyTorch.
* **File to study:** [`model.py` in llama2.c](https://github.com/karpathy/llama2.c/blob/master/model.py)
* **Why look at it:** It is the gold standard for readable, educational PyTorch code. If you get stuck on `rmsnorm.py` or RoPE, look here.

### 2. `nanoGPT` (by Andrej Karpathy)
This is for building a GPT-2 style model (older architecture using LayerNorm and standard Multi-Head Attention), but the training loops and dataset code are incredibly clean.
* **Repo:** [nanoGPT](https://github.com/karpathy/nanoGPT)
* **Why look at it:** It's the best reference for how to write a simple, fast training loop (`train.py`) and load data efficiently.

### 3. Lit-GPT (by Lightning AI)
A more production-ready but still highly readable implementation of modern LLMs (LLaMA, Mistral, etc.) without the bloated abstractions of HuggingFace.
* **Repo:** [lit-gpt](https://github.com/Lightning-AI/lit-gpt)
* **Why look at it:** Great for seeing how to implement Grouped Query Attention (GQA) and KV-caching efficiently.

### 4. TinyLlama
A project that pre-trained a 1.1B LLaMA model on 3 Trillion tokens. 
* **Repo:** [TinyLlama](https://github.com/jzhang38/TinyLlama)
* **Why look at it:** Great reference for how pre-training is actually orchestrated at scale (data packing, learning rate schedules, etc.).
