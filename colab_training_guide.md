# Colab LLM Training Guide (>100M Parameters)

You are aiming to build a >100M parameter model (which aligns perfectly with our 124M "Small" config) and train it on the datasets that powered GPT-2 and LLaMA using Google Colab. 

Training an LLM of this size in Colab is entirely possible, but you must be incredibly strategic about **Dataset Management** and **VRAM (GPU Memory) Optimization**. Colab instances (especially free ones) have strict memory limits and can disconnect.

Here is your exact step-by-step playbook to make this happen.

---

## 1. The Datasets: GPT-2 and LLaMA Reproductions

The original datasets for GPT-2 (WebText) and LLaMA are not publicly available exactly as they were used, but the open-source community has created near-perfect reproductions that you can easily download via HuggingFace.

### For the GPT-2 Experience: **OpenWebText**
OpenWebText is an open-source recreation of the WebText dataset used to train GPT-2. It consists of millions of outbound Reddit links with high karma.
- **Size:** ~38GB of raw text.
- **HuggingFace ID:** `Skylion007/openwebtext`

### For the LLaMA Experience: **RedPajama** (or FineWeb)
LLaMA was trained on a massive mixture of CommonCrawl, Wikipedia, Books, ArXiv, and GitHub. The community reproduced this as **RedPajama**.
- **Size:** 1.2 Trillion tokens (way too big for Colab, but we will use a sample!).
- **HuggingFace ID:** `togethercomputer/RedPajama-Data-1T-Sample` (A 1-billion token sample perfect for Colab training).

> [!TIP]
> **My Recommendation:** Start with the **RedPajama 1B token sample** or **FineWeb-Edu (Sample)**. They are highly curated, modern, and mimic LLaMA's data mixture perfectly, while being small enough to actually process in Colab.

---

## 2. Preparing the Data for Colab (CRITICAL)

You **cannot** download raw text files and tokenize them on-the-fly during training in Colab. It will crash the RAM and starve the GPU. You must pre-tokenize the data.

**What you should do (in a local script or a separate Colab data-prep notebook):**
1. Write a script to stream the dataset from HuggingFace.
2. Tokenize the text using your BPE tokenizer (or `tiktoken` / `AutoTokenizer`).
3. Save the token IDs as a single, massive 1D array of integers in a binary file using `numpy.memmap` (e.g., `train.bin` and `val.bin`).
4. **Upload these `.bin` files to Google Drive.**

During training, your Colab notebook will just mount Google Drive, read the binary file via `np.memmap` (which uses zero RAM, reading straight from disk/drive), and slice off chunks of `block_size` for your batches.

---

## 3. Model Architecture Sizing

A >100M parameter model fits perfectly into the classic **124M architecture** (GPT-2 Small size). Since we are using modern LLaMA techniques (RMSNorm, RoPE, SwiGLU, GQA) as outlined in your `implementation_plan.md`, this is the config you should use:

```python
vocab_size = 50257      # Standard GPT-2 vocab size
n_layer = 12            # 12 Transformer blocks
n_head = 12             # 12 Query attention heads
n_kv_head = 4           # 4 KV heads for Grouped Query Attention (saves memory!)
n_embd = 768            # Hidden dimension size
block_size = 1024       # Context window (tokens)
```
*Total Parameters: ~124 Million*

---

## 4. Colab Training Strategy & VRAM Optimization

A free Colab gives you an NVIDIA T4 GPU with 16GB of VRAM. A 124M parameter model takes about `500MB` just to store the weights. However, the gradients, optimizer states (Adam stores 2x the model size), and forward activations can quickly blow up to 10GB+. 

Here is exactly how you write your training loop to survive in Colab:

### A. Mount Google Drive
Colab instances wipe when they disconnect. You must save your checkpoints to Google Drive.
```python
from google.colab import drive
drive.mount('/content/drive')
# Save checkpoints to: /content/drive/MyDrive/ONE-llm/checkpoints/
```

### B. Use Mixed Precision (Automatic Mixed Precision - AMP)
You must train using 16-bit floats for the forward/backward passes, while keeping the optimizer states in 32-bit.
```python
import torch

# If Colab assigns you an A100/V100/L4, use bfloat16. If T4, use float16.
dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Inside your training loop:
with torch.autocast(device_type='cuda', dtype=ptdtype):
    logits, loss = model(x, y)
```

### C. Gradient Accumulation
You probably won't be able to fit a batch size of more than 8 or 16 in a T4 GPU. To simulate a large batch size (which LLMs need for stable training), you accumulate gradients over multiple micro-steps before running the optimizer step.
```python
gradient_accumulation_steps = 8
micro_batch_size = 8
# Effective batch size = 8 * 8 = 64

for i, (X, Y) in enumerate(dataloader):
    with torch.autocast(device_type='cuda', dtype=ptdtype):
        logits, loss = model(X, Y)
        # Scale loss to account for accumulation
        loss = loss / gradient_accumulation_steps 
    
    loss.backward()
    
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### D. Use `torch.compile()`
PyTorch 2.0 introduced `torch.compile()`, which fuses operations and drastically speeds up training (up to 30% faster on modern GPUs).
```python
model = ONE_LLM(config).to(device)
model = torch.compile(model) # Magic speedup line
```

---

## Next Steps for You

1. **Do you agree with using the RedPajama 1B sample dataset to mimic LLaMA?**
2. **Do you want me to write the `data/dataset.py` script right now so you can prepare the `.bin` files?** (This is the mandatory first step before we can run any training loop).
