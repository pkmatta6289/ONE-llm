# ONE-LLM Mathematical Foundations

This document explains the mathematical formulas behind the modern transformer components we are building.

---

## 1. Root Mean Square Normalization (RMSNorm)

**Goal:** Normalize activations to stabilize training, but do it faster than standard LayerNorm.

**Standard LayerNorm Math:**
1. Mean: $\mu = \frac{1}{d} \sum_{i=1}^d x_i$
2. Variance: $\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$
3. Output: $y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$

**RMSNorm Math:**
Researchers found that the mean-centering ($\mu$) in LayerNorm isn't strictly necessary for success. If we drop it, we get RMSNorm:
1. Root Mean Square: $\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2}$
2. Output: $y = \frac{x}{\text{RMS}(x) + \epsilon} \cdot \gamma$

*Where:*
- $d$ is the hidden dimension (`dim`).
- $\epsilon$ (`eps`) is a tiny number (e.g., $1e-5$) to prevent division by zero.
- $\gamma$ (`weight`) is a learnable scaling parameter of size $d$.

**Why it's better:** It saves computing the mean and subtracting it from every element, resulting in a ~10-40% speedup in the normalization layer with no drop in model accuracy.

---

## 2. Rotary Position Embeddings (RoPE)

**Goal:** Inject positional information into the sequence so the model knows the order of tokens. RoPE uses rotations in complex space to encode relative distances between tokens perfectly.

**The Math:**
Imagine the query vector $q$ at position $m$ and the key vector $k$ at position $n$. We want their dot product $q_m \cdot k_n$ to depend only on their relative distance $(m - n)$.

1. We pair up the features in the hidden dimension: $(x_0, x_1), (x_2, x_3), \dots$
2. We treat each pair as a complex number: $x_0 + i x_1$
3. For a specific feature pair $j$, we define a frequency: $\theta_j = 10000^{-2j/d}$
4. We rotate the complex number by an angle proportional to its position $m$: 
   $e^{i m \theta_j} = \cos(m \theta_j) + i \sin(m \theta_j)$
5. The rotated feature is: $(x_0 + i x_1) \cdot e^{i m \theta_j}$

**The Dot Product Magic:**
When we take the dot product of a rotated query and a rotated key, the absolute positions $m$ and $n$ cancel out in a way that leaves only their difference:
$\langle \text{RoPE}(q, m), \text{RoPE}(k, n) \rangle = \text{function}(q, k, m - n)$

**Why it's better:** It seamlessly extends to longer context lengths and naturally captures relative distances (e.g., "token A is 3 steps away from token B") which is how language works.

---

## 3. SwiGLU Feed-Forward Network (FFN)

**Goal:** Process information within each token's representation.

**Standard Transformer FFN:**
Uses two linear layers with a ReLU (or GELU) in the middle.
$\text{FFN}(x) = \text{ReLU}(x \cdot W_1) \cdot W_2$

**SwiGLU Math:**
Instead of just activating the linear projection, SwiGLU uses a gating mechanism where two linear projections are multiplied together, and one passes through a SiLU (Swish) activation function.
1. SiLU activation: $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$
2. SwiGLU output: $y = \left( \text{SiLU}(x \cdot W_{\text{gate}}) \otimes (x \cdot W_{\text{up}}) \right) \cdot W_{\text{down}}$

*Where:*
- $\otimes$ is element-wise multiplication.
- $W_{\text{gate}}$ and $W_{\text{up}}$ project from `dim` to `hidden_dim`.
- $W_{\text{down}}$ projects back from `hidden_dim` to `dim`.

**Why it's better:** The multiplicative gating acts as an information filter, and empirical results show it significantly outperforms standard ReLU/GELU FFNs.

---

## 4. Grouped Query Attention (GQA)

**Goal:** Reduce the memory footprint of the KV-cache during generation (inference) without hurting accuracy.

**Multi-Head Attention (MHA - e.g., GPT-2):**
If `n_heads = 12`, we compute 12 Query heads, 12 Key heads, and 12 Value heads.
During generation, we have to keep all 12 Key and Value heads in memory (the KV-cache). This uses massive amounts of RAM for large batches or long contexts.

**Multi-Query Attention (MQA):**
Compute 12 Query heads, but only 1 Key head and 1 Value head. All 12 Q heads share the same K and V.
This saves 12x memory, but hurts model performance and learning capacity.

**Grouped Query Attention (GQA - e.g., LLaMA, ONE-LLM):**
The golden middle ground. 
If `n_heads = 12`, we compute `n_kv_heads = 4` Key and Value heads.
We divide the 12 Q heads into 4 groups (3 heads per group). Each group shares one K and one V head.

**The Math (in PyTorch):**
We just repeat the K and V tensors before the dot product:
1. $K$ shape is `(batch, seq, 4, head_dim)`
2. Repeat K 3 times: `K_repeated` shape becomes `(batch, seq, 12, head_dim)`
3. Standard Attention: $\text{Softmax}\left(\frac{Q \cdot K_{\text{repeated}}^T}{\sqrt{\text{head\_dim}}}\right) \cdot V_{\text{repeated}}$

**Why it's better:** Provides ~95% of the memory savings of MQA while maintaining ~99% of the accuracy of standard MHA.
