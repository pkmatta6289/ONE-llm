import torch

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (RoPE).
    
    Args:
        dim: dimension of the embeddings (must be even, usually head_dim)
        end: maximum sequence length
        theta: base for the frequencies
    """
    # 1. Compute frequencies: 1 / (theta ** (2i / dim)) for i in range(0, dim, 2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    
    # 2. Create position indices (sequence length)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    
    # 3. Outer product of positions and frequencies: t * freqs
    freqs = torch.outer(t, freqs)
    
    # 4. Convert to complex numbers in polar form: r=1, theta=freqs
    # e^{i * theta} = cos(theta) + i * sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # shape (end, dim // 2)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to queries and keys.
    xq: (batch, seq_len, n_heads, head_dim)
    xk: (batch, seq_len, n_kv_heads, head_dim)
    freqs_cis: (seq_len, head_dim // 2)
    """
    # 1. Reshape xq and xk to view pairs of elements as complex numbers
    # We go from (..., head_dim) to (..., head_dim // 2, 2) and treat it as a complex tensor
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Reshape freqs_cis for broadcasting
    # freqs_cis comes in as (seq_len, head_dim // 2)
    # We need (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq.shape[1], 1, xq.shape[-1] // 2)
    
    # 3. Multiply and flatten back out
    # Multiplication in complex space performs the rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
